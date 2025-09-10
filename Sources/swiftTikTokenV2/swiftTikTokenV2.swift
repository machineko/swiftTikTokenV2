import Foundation
import Metal
import MetalPerformanceShaders

public typealias Rank = UInt32
let AUDIO_SAMPLES_PER_TOKEN: Float = 16000.0 * 0.02
let HOP_LENGTH: Float = 160
let SAMPLE_RATE: Float = 16000
let TOKENS_PER_SECOND: Float = 50.0  // 20ms

extension Set where Element == Character {
    func compatibleContains(_ string: String) -> Bool {
        if string.isEmpty { return false }
        return string.contains { char in
            if #available(macOS 13.0, *) {
                return self.contains(char)
            } else {
                return self.contains { $0 == char }
            }
        }
    }
}

extension Data: @retroactive Comparable {
    public static func < (lhs: Data, rhs: Data) -> Bool {
        let minLength = Swift.min(lhs.count, rhs.count)
        for i in 0..<minLength {
            if lhs[i] != rhs[i] {
                return lhs[i] < rhs[i]
            }
        }
        return lhs.count < rhs.count
    }
}

protocol RegexCompatibility {
    func matches(_ string: String) -> Bool
    func firstMatch(in string: String) -> (String, Range<String.Index>)?
    func allMatches(in string: String) -> [(String, Range<String.Index>)]
}

@available(macOS 13.0, *)
public class ModernRegex: RegexCompatibility {
    private let regex: Regex<Substring>

    init(_ pattern: String) throws {
        self.regex = try Regex(pattern)
    }

    func matches(_ string: String) -> Bool {
        string.contains(regex)
    }

    func firstMatch(in string: String) -> (String, Range<String.Index>)? {
        guard let match = string.firstMatch(of: regex) else { return nil }
        return (String(match.0), match.range)
    }

    func allMatches(in string: String) -> [(String, Range<String.Index>)] {
        string.matches(of: regex).map { (String($0.0), $0.range) }
    }
}

public class LegacyRegex: RegexCompatibility {
    private let regex: NSRegularExpression

    init(_ pattern: String) throws {
        self.regex = try NSRegularExpression(pattern: pattern, options: [])
    }

    func matches(_ string: String) -> Bool {
        let range = NSRange(string.startIndex..., in: string)
        return regex.firstMatch(in: string, options: [], range: range) != nil
    }

    func firstMatch(in string: String) -> (String, Range<String.Index>)? {
        let range = NSRange(string.startIndex..., in: string)
        guard let match = regex.firstMatch(in: string, options: [], range: range),
            let matchRange = Range(match.range, in: string)
        else {
            return nil
        }
        return (String(string[matchRange]), matchRange)
    }

    func allMatches(in string: String) -> [(String, Range<String.Index>)] {
        let range = NSRange(string.startIndex..., in: string)
        return regex.matches(in: string, options: [], range: range)
            .compactMap { match -> (String, Range<String.Index>)? in
                guard let matchRange = Range(match.range, in: string) else { return nil }
                return (String(string[matchRange]), matchRange)
            }
    }
}

public class RegexFactory {
    static func createRegex(_ pattern: String) throws -> RegexCompatibility {
        if #available(macOS 13.0, *) {
            return try ModernRegex(pattern)
        } else {
            return try LegacyRegex(pattern)
        }
    }
}

public struct CoreBPE {
    private let encoder: [Data: Rank]
    private let specialTokensEncoder: [String: Rank]
    private let decoder: [Rank: Data]
    private let specialTokensDecoder: [Rank: Data]
    private let regex: RegexCompatibility
    private let specialRegex: RegexCompatibility
    private let sortedTokenBytes: [Data]

    init(encoder: [Data: Rank], specialTokensEncoder: [String: Rank], pattern: String) throws {
        self.encoder = encoder
        self.specialTokensEncoder = specialTokensEncoder
        self.decoder = Dictionary(uniqueKeysWithValues: encoder.map { ($1, $0) })
        self.specialTokensDecoder = Dictionary(
            uniqueKeysWithValues: specialTokensEncoder.map { ($1, $0.data(using: .utf8)!) }
        )

        self.regex = try RegexFactory.createRegex(pattern)

        let sortedSpecialTokens = specialTokensEncoder.keys.sorted { $0.count > $1.count }
        let specialPattern =
            sortedSpecialTokens
            .map {
                NSRegularExpression.escapedPattern(for: $0)
            }
            .joined(separator: "|")
        self.specialRegex = try RegexFactory.createRegex(specialPattern)

        self.sortedTokenBytes = encoder.keys.sorted()
    }

    private func bytePairMerge(_ ranks: [Data: Rank], piece: Data) -> [(Int, Rank)] {
        var parts: [(Int, Rank)] = []
        var minRank: (rank: Rank, index: Int) = (Rank.max, Int.max)

        for i in 0..<(piece.count - 1) {
            let subdata = piece.subdata(in: i..<(i + 2))
            let rank = ranks[subdata] ?? Rank.max
            if rank < minRank.rank {
                minRank = (rank, i)
            }
            parts.append((i, rank))
        }

        parts.append((piece.count - 1, Rank.max))
        parts.append((piece.count, Rank.max))

        while minRank.rank != Rank.max {
            let i = minRank.index

            if i > 0 {
                parts[i - 1].1 = getRank(parts: parts, piece: piece, ranks: ranks, index: i - 1)
            }
            parts[i].1 = getRank(parts: parts, piece: piece, ranks: ranks, index: i)
            parts.remove(at: i + 1)

            minRank = (Rank.max, Int.max)
            for (index, part) in parts[..<(parts.count - 1)].enumerated() {
                if part.1 < minRank.rank {
                    minRank = (part.1, index)
                }
            }
        }

        return parts
    }

    private func getRank(parts: [(Int, Rank)], piece: Data, ranks: [Data: Rank], index: Int) -> Rank {
        if (index + 3) < parts.count {
            let subdata = piece.subdata(in: parts[index].0..<parts[index + 3].0)
            return ranks[subdata] ?? Rank.max
        }
        return Rank.max
    }

    private static func bytesToUnicode() -> [UInt8: UInt8] {
        var bs: [UInt8] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs: [UInt8] = bs.map { $0 }

        var n: UInt8 = 0
        for b in 0...255 {
            if !bs.contains(UInt8(b)) {
                bs.append(UInt8(b))
                cs.append(UInt8(n))
                n += 1
            }
        }

        return Dictionary(uniqueKeysWithValues: zip(bs, cs))
    }

    func bytePairEncode(piece: Data) -> [Rank] {
        // For single bytes, encode directly
        if piece.count == 1 {
            guard let token = encoder[piece] else {
                // For space character, use a specific token if available
                if piece[0] == 32, let spaceToken = specialTokensEncoder[" "] {
                    return [spaceToken]
                }
                fatalError("Unable to encode single character: \(piece[0])")
            }
            return [token]
        }

        // Split on spaces and encode each part separately
        var tokens: [Rank] = []
        var currentPiece = Data()

        for byte in piece {
            if byte == 32 {  // Space character
                // Encode current piece if not empty
                if !currentPiece.isEmpty {
                    let parts = bytePairMerge(encoder, piece: currentPiece)
                    tokens.append(
                        contentsOf: parts.windows(ofCount: 2).compactMap { window in
                            let start = window[0].0
                            let end = window[1].0
                            return encoder[currentPiece.subdata(in: start..<end)]
                        })
                    currentPiece = Data()
                }

                // Add space token
                if let spaceToken = encoder[Data([32])] {
                    tokens.append(spaceToken)
                } else if let spaceToken = specialTokensEncoder[" "] {
                    tokens.append(spaceToken)
                } else {
                    fatalError("Space character not found in encoder or special tokens")
                }
            } else {
                currentPiece.append(byte)
            }
        }

        if !currentPiece.isEmpty {
            let parts = bytePairMerge(encoder, piece: currentPiece)
            tokens.append(
                contentsOf: parts.windows(ofCount: 2).compactMap { window in
                    let start = window[0].0
                    let end = window[1].0
                    return encoder[currentPiece.subdata(in: start..<end)]
                })
        }

        return tokens
    }

    func encode(_ text: String, allowedSpecial: Set<String>) -> [Rank] {
        var tokens: [Rank] = []
        var currentIndex = text.startIndex

        while currentIndex < text.endIndex {
            let remainingText = String(text[currentIndex...])

            if let specialMatch = specialRegex.firstMatch(in: remainingText) {
                let matchStart = specialMatch.1.lowerBound
                if currentIndex < text.index(currentIndex, offsetBy: matchStart.utf16Offset(in: remainingText)) {
                    let beforeSpecial = String(
                        text[currentIndex..<text.index(currentIndex, offsetBy: matchStart.utf16Offset(in: remainingText))])
                    for match in regex.allMatches(in: beforeSpecial) {
                        if let piece = text[match.1].data(using: .utf8) {
                            tokens.append(contentsOf: bytePairEncode(piece: piece))
                        }
                    }
                }

                let specialToken = String(remainingText[specialMatch.1])
                if allowedSpecial.contains(specialToken) {
                    if let specialRank = specialTokensEncoder[specialToken] {
                        tokens.append(specialRank)
                    }
                } else {
                    // If special token is not allowed, encode it as regular text
                    if let piece = specialToken.data(using: .utf8) {
                        tokens.append(contentsOf: bytePairEncode(piece: piece))
                    }
                }

                currentIndex = text.index(currentIndex, offsetBy: specialMatch.1.upperBound.utf16Offset(in: remainingText))
            } else {
                for match in regex.allMatches(in: remainingText) {
                    if let piece = text[match.1].data(using: .utf8) {
                        tokens.append(contentsOf: bytePairEncode(piece: piece))
                    }
                }
                break
            }
        }

        return tokens
    }

    func encode(_ text: String) -> [Rank] {
        var tokens: [Rank] = []
        var currentIndex = text.startIndex

        while currentIndex < text.endIndex {
            let remainingText = String(text[currentIndex...])

            if let specialMatch = specialRegex.firstMatch(in: remainingText) {
                let matchStart = specialMatch.1.lowerBound

                if currentIndex < text.index(currentIndex, offsetBy: matchStart.utf16Offset(in: remainingText)) {
                    let beforeSpecial = String(
                        text[currentIndex..<text.index(currentIndex, offsetBy: matchStart.utf16Offset(in: remainingText))])
                    for match in regex.allMatches(in: beforeSpecial) {
                        if let piece = text[match.1].data(using: .utf8) {
                            tokens.append(contentsOf: bytePairEncode(piece: piece))
                        }
                    }
                }

                let specialToken = String(remainingText[specialMatch.1])
                if let specialRank = specialTokensEncoder[specialToken] {
                    tokens.append(specialRank)
                } else {
                    fatalError("token \(specialToken) not supported")
                }

                currentIndex = text.index(currentIndex, offsetBy: specialMatch.1.upperBound.utf16Offset(in: remainingText))
            } else {
                for match in regex.allMatches(in: remainingText) {
                    if let piece = text[match.1].data(using: .utf8) {
                        tokens.append(contentsOf: bytePairEncode(piece: piece))
                    }
                }
                break
            }
        }

        return tokens
    }

    func decode(_ tokens: [Rank]) throws -> Data {
        var result = Data()
        for token in tokens {
            if let bytes = decoder[token] {
                result.append(bytes)
            } else if let bytes = specialTokensDecoder[token] {
                result.append(bytes)
            } else {
                throw TokenizerError.invalidToken(token)
            }
        }
        return result
    }
}

enum BPEError: Error {
    case invalidToken(Rank)
    case invalidPattern(String)
}

extension Array {
    func windows(ofCount count: Int) -> [[Element]] {
        guard count > 0, self.count >= count else { return [] }
        return (0...self.count - count).map {
            Array(self[$0..<$0 + count])
        }
    }
}

struct GPT2Encoder {
    static func get() throws -> CoreBPE {
        let (encoder, specialTokens, pattern) = try loadGPT2Parameters()
        return try CoreBPE(
            encoder: encoder,
            specialTokensEncoder: specialTokens,
            pattern: pattern)
    }

    private static func loadGPT2Parameters() throws -> ([Data: Rank], [String: Rank], String) {
        // GPT-2 regex pattern
        let pattern = #"""
            \'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
            """#

        // Special tokens for GPT-2
        let specialTokens: [String: Rank] = [
            "<|endoftext|>": 50256
        ]

        let url = Bundle.module.url(forResource: "gpt2encoder", withExtension: "json", subdirectory: "Resources")!
        let data = try! Data(contentsOf: url)
        let encoderDict = try! JSONSerialization.jsonObject(with: data) as! [String: Int]
        let byteEncoder = bytesToUnicode()

        var encoder: [Data: Rank] = [:]
        for (token, rank) in encoderDict {
            if let tokenData = token.data(using: .utf8) {
                let encodedBytes = tokenData.map { byteEncoder[$0]! }
                encoder[Data(encodedBytes)] = Rank(rank)
            }
        }

        return (encoder, specialTokens, pattern)
    }

    private static func bytesToUnicode() -> [UInt8: UInt8] {
        var bs: [UInt8] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs: [UInt8] = bs.map { $0 }

        var n: UInt8 = 0
        for b in 0...255 {
            if !bs.contains(UInt8(b)) {
                bs.append(UInt8(b))
                cs.append(UInt8(n))
                n += 1
            }
        }

        return Dictionary(uniqueKeysWithValues: zip(bs, cs))
    }
}

class Tokenizer {
    private let bpe: CoreBPE

    init() throws {
        self.bpe = try GPT2Encoder.get()
    }

    func encode(_ text: String) -> [Rank] {
        bpe.encode(text, allowedSpecial: ["<|endoftext|>"])
    }

    func decode(_ tokens: [Rank]) -> String {
        guard let decoded = try? bpe.decode(tokens),
            let text = String(data: decoded, encoding: .utf8)
        else {
            return ""
        }
        return text
    }
}

struct MultilingualTokenizer {
    private let bpe: CoreBPE

    init() throws {
        self.bpe = try MultilingualEncoder.get()
    }

    func encode(_ text: String) -> [Rank] {
        bpe.encode(text, allowedSpecial: ["<|endoftext|>", "<|startoftranscript|>"])
    }

    func encode(_ text: String, allowAllSpecials: Bool, allowedSpecial: Set<String> = []) -> [Rank] {
        if allowAllSpecials {
            return bpe.encode(text)
        }
        return bpe.encode(text, allowedSpecial: allowedSpecial)
    }

    func decode(_ tokens: [Rank]) -> String {
        guard let decoded = try? bpe.decode(tokens),
            let text = String(data: decoded, encoding: .utf8)
        else {
            return ""
        }
        return text
    }
}

public struct MultilingualEncoder {
    public static func get() throws -> CoreBPE {
        let (encoder, specialTokens, pattern) = try loadMultilingualParameters()
        return try CoreBPE(
            encoder: encoder,
            specialTokensEncoder: specialTokens,
            pattern: pattern)
    }

    private static func loadMultilingualParameters() throws -> ([Data: Rank], [String: Rank], String) {
        let url = Bundle.module.url(forResource: "whisper_tokenizer", withExtension: "json", subdirectory: "Resources")!
        let data = try Data(contentsOf: url)

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let encoderDict = json["encoder"] as? [String: Int],
            let specialTokensDict = json["special_tokens"] as? [String: Int],
            let pattern = json["pattern"] as? String
        else {
            throw TokenizerError.invalidJSON("Invalid tokenizer format")
        }

        var encoder: [Data: Rank] = [:]
        for (hexToken, rank) in encoderDict {
            if let data = Data(hexString: hexToken) {
                encoder[data] = Rank(rank)
            }
        }

        var specialTokens: [String: Rank] = [:]
        for (token, rank) in specialTokensDict {
            specialTokens[token] = Rank(rank)
        }

        let uniqueRanks = Set(specialTokens.values)
        if uniqueRanks.count != specialTokens.count {
            throw TokenizerError.invalidVocabulary("Duplicate ranks found in special tokens")
        }

        return (encoder, specialTokens, pattern)
    }
}

extension Data {
    init?(hexString: String) {
        let len = hexString.count / 2
        var data = Data(capacity: len)
        var index = hexString.startIndex
        for _ in 0..<len {
            let nextIndex = hexString.index(index, offsetBy: 2)
            let bytes = hexString[index..<nextIndex]
            if let byte = UInt8(bytes, radix: 16) {
                data.append(byte)
            } else {
                return nil
            }
            index = nextIndex
        }
        self = data
    }
}

enum TokenizerError: LocalizedError {
    case encoderNotFound
    case invalidLanguage(String)
    case invalidTask(String)
    case encodingError
    case decodingError
    case languageMapNotFound
    case invalidLanguageMap
    case invalidVocabulary(String)
    case fileNotFound(String)
    case invalidJSON(String)
    case specialTokenNotFound(String)
    case invalidToken(Rank)
    case disallowedSpecialToken(String)

    var errorDescription: String? {
        switch self {
        case .encoderNotFound:
            return "Encoder vocabulary file not found"
        case .invalidLanguage(let lang):
            return "Unsupported language: \(lang)"
        case .invalidTask(let task):
            return "Invalid task: \(task). Must be 'transcribe' or 'translate'"
        case .encodingError:
            return "Failed to encode text"
        case .decodingError:
            return "Failed to decode tokens"
        case .languageMapNotFound:
            return "Language mapping file not found"
        case .invalidLanguageMap:
            return "Invalid language mapping format"
        case .invalidVocabulary(let val):
            return "Invalid vocabulary format \(val)"
        case .fileNotFound(let filename):
            return "File not found: \(filename)"
        case .invalidJSON(let details):
            return "Invalid JSON format: \(details)"
        case .specialTokenNotFound(let token):
            return "specialTokenNotFound: \(token)"
        case .invalidToken(let token):
            return "Invalid token \(token)"
        case .disallowedSpecialToken(let token):
            return "Encountered disallowed special token: \(token)"

        }
    }
}

struct Languages {
    private static let languageData: LanguageData = {
        guard let url = Bundle.module.url(forResource: "whisper_languages", withExtension: "json", subdirectory: "Resources"),
            let data = try? Data(contentsOf: url),
            let decoded = try? JSONDecoder().decode(LanguageData.self, from: data)
        else {
            fatalError("Failed to load languages.json")
        }
        return decoded
    }()

    static var codes: [String: String] { languageData.codes }
    static var toLanguageCode: [String: String] { languageData.aliases }

    private init() {}
}

struct LanguageData: Codable {
    let codes: [String: String]
    let aliases: [String: String]
}

@available(macOS 13.0, *)
extension CoreBPE {
    public func debugToken(_ data: Data) {
        print("Debugging token:", data.map { String(format: "%02x", $0) }.joined())
        if let rank = encoder[data] {
            print("Found rank:", rank)
        } else {
            print("Token not found in encoder")
            print("Available similar tokens:")
            for (tokenData, rank) in encoder where tokenData.count == data.count {
                print(tokenData.map { String(format: "%02x", $0) }.joined(), "->", rank)
            }
        }
    }
}

public class WhisperTokenizer {
    public var cache: [String: Any] = [:]
    public let bpe: CoreBPE
    public let numLanguages: Int
    public var language: String?
    public let task: String?
    public var sotSequence: [Rank] = []
    public var specialTokens: [String: Rank]
    public lazy var timestampBegin: Rank = {
        specialTokens["<|0.00|>"] ?? 0
    }()

    public enum SpecialTokensOption {
        case all
        case specific(Set<String>)
        case none

        public var asSet: Set<String>? {
            switch self {
            case .all: return nil
            case .specific(let set): return set
            case .none: return []
            }
        }
    }

    public init(bpe: CoreBPE, numLanguages: Int = 100, language: String? = nil, task: String? = nil) throws {
        self.bpe = bpe
        self.numLanguages = numLanguages
        self.language = language
        self.task = task

        if let specialTokensEncoder = Mirror(reflecting: bpe).children
            .first(where: { $0.label == "specialTokensEncoder" })?.value as? [String: Rank]
        {
            self.specialTokens = specialTokensEncoder
        } else {
            throw TokenizerError.specialTokenNotFound("Could not access specialTokensEncoder")
        }

        try setupSotSequence()
    }

    public func setupSotSequence() throws {
        guard let sot = specialTokens["<|startoftranscript|>"],
            let translate = specialTokens["<|translate|>"],
            let transcribe = specialTokens["<|transcribe|>"]
        else {
            throw TokenizerError.specialTokenNotFound("Required special tokens not found")
        }

        var sequence: [Rank] = [sot]

        if let language = language {
            if let langToken = specialTokens["<|\(language)|>"] {
                sequence.append(langToken)
            }
        }

        if let task = task {
            let taskToken = task == "transcribe" ? transcribe : translate
            sequence.append(taskToken)
        }

        sotSequence = sequence
    }
    public func setupSpecialTokens() throws {
        let specialTokensList = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
        ]

        for special in specialTokensList {
            if let token = encode(special).first {
                specialTokens[special] = token
            }
        }

        for i in 0...1500 {
            let timestamp = String(format: "<|%.2f|>", Double(i) * 0.02)
            if let token = encode(timestamp).first {
                specialTokens[timestamp] = token
            }
        }
    }

    //    func encode(_ text: String) throws -> [Rank] {
    //        bpe.encode(text, allowedSpecial: Set(specialTokens.keys))
    //    }

    public func encode(_ text: String) -> [Rank] {
        bpe.encode(text, allowedSpecial: Set(specialTokens.keys))
    }

    public func encode(_ text: String, allowAllSpecials: Bool, allowedSpecial: Set<String> = []) -> [Rank] {
        if allowAllSpecials {
            return bpe.encode(text)
        }
        return bpe.encode(text, allowedSpecial: allowedSpecial)
    }

    public func decode(_ tokens: [Rank]) -> String {
        let filteredTokens = tokens.filter { $0 < timestampBegin }
        return (try? bpe.decode(filteredTokens)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }

    public func decodeWithTimestamps(_ tokens: [Rank]) -> String {
        return (try? bpe.decode(tokens)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }

    public var eot: Rank { specialTokens["<|endoftext|>"] ?? 0 }
    public var transcribe: Rank { specialTokens["<|transcribe|>"] ?? 0 }
    public var translate: Rank { specialTokens["<|translate|>"] ?? 0 }
    public var sot: Rank { specialTokens["<|startoftranscript|>"] ?? 0 }
    public var sotLm: Rank { specialTokens["<|startoflm|>"] ?? 0 }
    public var sotPrev: Rank { specialTokens["<|startofprev|>"] ?? 0 }
    public var noSpeech: Rank { specialTokens["<|nospeech|>"] ?? 0 }
    public var noTimestamps: Rank { specialTokens["<|notimestamps|>"] ?? 0 }

    public var languageToken: Rank {
        get throws {
            guard let language = language,
                let token = specialTokens["<|\(language)|>"]
            else {
                throw TokenizerError.invalidLanguage("Language token not found")
            }
            return token
        }
    }

    public func toLanguageToken(_ language: String) throws -> Rank {
        guard let token = specialTokens["<|\(language)|>"] else {
            throw TokenizerError.invalidLanguage("Language \(language) not found")
        }
        return token
    }

    public var allLanguageTokens: [Rank] {
        var result: [Rank] = []
        for (token, tokenId) in specialTokens {
            let stripped = token.trimmingCharacters(in: CharacterSet(charactersIn: "<|>"))
            if Languages.codes.keys.contains(stripped) {
                result.append(tokenId)
            }
        }
        return Array(result.prefix(numLanguages))
    }

    public var allLanguageCodes: [String] {
        return allLanguageTokens.compactMap { token in
            guard let decoded = try? bpe.decode([token]),
                let str = String(data: decoded, encoding: .utf8)
            else {
                return nil
            }
            return str.trimmingCharacters(in: CharacterSet(charactersIn: "<|>"))
        }
    }

    public var sotSequenceIncludingNoTimestamps: [Rank] {
        return sotSequence + [noTimestamps]
    }
}

extension WhisperTokenizer {
    var nonSpeechTokens: Set<Rank> {
        var result = Set<Rank>()

        let symbols = [
            "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "「", "」",
            "『", "』",
        ]

        let multiTokenSymbols = "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split(separator: " ").map(String.init)

        let miscellaneous: Set<String> = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"]

        if let hyphenToken = encode(" -").first {
            result.insert(hyphenToken)
        }
        if let quoteToken = encode(" '").first {
            result.insert(quoteToken)
        }

        let allSymbols = symbols + multiTokenSymbols + Array(miscellaneous)
        for symbol in allSymbols {

            let tokens = encode(symbol)
            result.insert(tokens.first!)

            let tokensWithSpace = encode(" " + symbol)
            result.insert(tokensWithSpace.first!)
        }

        return result
    }
}

extension WhisperTokenizer {
    func splitToWordTokens(_ tokens: [Rank]) -> ([String], [[Rank]]) {
        let nonSpaceLanguages = Set(["zh", "ja", "th", "lo", "my", "yue"])

        if let language = language, nonSpaceLanguages.contains(language) {
            return splitTokensOnUnicode(tokens)
        }

        return splitTokensOnSpaces(tokens)
    }

    private func splitTokensOnUnicode(_ tokens: [Rank]) -> ([String], [[Rank]]) {
        let decodedFull = decodeWithTimestamps(tokens)
        let replacementChar = "\u{FFFD}"

        var words: [String] = []
        var wordTokens: [[Rank]] = []
        var currentTokens: [Rank] = []
        var unicodeOffset = 0

        for token in tokens {
            currentTokens.append(token)
            let decoded = decodeWithTimestamps(currentTokens)

            if !decoded.contains(replacementChar)
                || decodedFull[
                    decodedFull.index(
                        decodedFull.startIndex,
                        offsetBy: unicodeOffset + decoded.firstIndex(of: Character(replacementChar))!.utf16Offset(in: decoded))]
                    == Character(replacementChar)
            {
                words.append(decoded)
                wordTokens.append(currentTokens)
                currentTokens = []
                unicodeOffset += decoded.count
            }
        }

        return (words, wordTokens)
    }

    private func splitTokensOnSpaces(_ tokens: [Rank]) -> ([String], [[Rank]]) {
        let (subwords, subwordTokensList) = splitTokensOnUnicode(tokens)
        var words: [String] = []
        var wordTokens: [[Rank]] = []

        for (_, (subword, subwordTokens)) in zip(subwords, subwordTokensList).enumerated() {
            let special = subwordTokens[0] >= eot
            let withSpace = subword.hasPrefix(" ")
            let punctuation = subword.trimmingCharacters(in: .whitespaces).allSatisfy {
                CharacterSet.punctuationCharacters.contains($0.unicodeScalars.first!)
            }

            if special || withSpace || punctuation || words.isEmpty {
                words.append(subword)
                wordTokens.append(subwordTokens)
            } else {
                words[words.count - 1] += subword
                wordTokens[wordTokens.count - 1].append(contentsOf: subwordTokens)
            }
        }

        return (words, wordTokens)
    }
}

extension WhisperTokenizer {

    private func cached<T>(_ key: String, compute: () -> T) -> T {
        if let cached = cache[key] as? T {
            return cached
        }
        let value = compute()
        cache[key] = value
        return value
    }
}

extension WhisperTokenizer {
    func encode(
        text: String,
        allowedSpecial: SpecialTokensOption = .specific(["<|endoftext|>"]),
        disallowedSpecial: SpecialTokensOption = .all
    ) throws -> [Rank] {
        let allowedTokens: Set<String> = {
            switch allowedSpecial {
            case .all:
                return Set(specialTokens.keys)
            case .specific(let tokens):
                return tokens
            case .none:
                return []
            }
        }()

        let disallowedTokens: Set<String> = {
            switch disallowedSpecial {
            case .all:
                return Set(specialTokens.keys).subtracting(allowedTokens)
            case .specific(let tokens):
                return tokens
            case .none:
                return []
            }
        }()

        if !disallowedTokens.isEmpty {
            let pattern =
                disallowedTokens
                .map { "\\Q\($0)\\E" }
                .joined(separator: "|")

            let regex = try RegexFactory.createRegex(pattern)
            if let match = regex.firstMatch(in: text) {
                throw TokenizerError.disallowedSpecialToken(String(text[match.1]))
            }
        }

        return bpe.encode(text, allowedSpecial: allowedTokens)
    }

    func prepareTranscriptionPrompt(prompt: String?, withoutSotSequence: Bool = false, removeTimestamps: Bool = true) throws -> [Rank] {
        var tokens: [Rank] = []
        if !withoutSotSequence {
            tokens.append(contentsOf: sotSequence)
        }
        if removeTimestamps {
            tokens.append(noTimestamps)
        }
        if let prompt = prompt {
            let promptTokens = self.encode(prompt)
            tokens.append(contentsOf: promptTokens)
        }

        return tokens
    }

    func decodeTranscription(_ tokens: [Rank], removeTimestamps: Bool = true) -> String {
        let processedTokens =
            tokens
            .prefix { $0 != eot }
            .filter { token in
                token != noSpeech && (!removeTimestamps || tokens.contains(token))
            }

        if removeTimestamps {
            return decode(processedTokens)
        } else {
            return decodeWithTimestamps(processedTokens)
        }
    }

    func isTimestamp(_ token: Rank) -> Bool {
        token >= timestampBegin
    }

    func filterNonSpeechTokens(_ tokens: [Rank]) -> [Rank] {
        tokens.filter { !nonSpeechTokens.contains($0) }
    }

    func prepareAlignmentSequence(
        transcription: String,
        durationInSeconds: Float
    ) throws -> [Rank] {
        var tokens: [Rank] = []

        tokens.append(contentsOf: sotSequence)

        tokens.append(timestampBegin)

        let transcriptionTokens = try encode(
            text: transcription,
            allowedSpecial: SpecialTokensOption.specific(["<|endoftext|>", "<|startoftranscript|>"])
        )
        tokens.append(contentsOf: transcriptionTokens)

        let finalTimestampOffset = Int(durationInSeconds / 0.02)
        tokens.append(timestampBegin + Rank(finalTimestampOffset))

        if let eotToken = specialTokens["<|endoftext|>"] {
            tokens.append(eotToken)
        }

        return tokens
    }

    func timestampToken(forTime seconds: Float) -> Rank? {
        let tokenIndex = Int(seconds / 0.02)
        return timestampBegin + Rank(tokenIndex)
    }
}

extension WhisperTokenizer {
    public struct WordTiming {
        var word: String
        var tokens: [Rank]
        var start: Float
        var end: Float
        let probability: Float
        let entropy: Float
    }

    public struct Segment: CustomStringConvertible, CustomDebugStringConvertible, Sendable {

        public var start: Float
        public var end: Float
        public var tokens: [Rank]
        public var words: [Word]
        public let seek: Int

        public struct Word: CustomStringConvertible, CustomDebugStringConvertible, Sendable {
            public let word: String
            public var start: Float
            public var end: Float
            public let probability: Float
            public let entropy: Float

            public var description: String {
                return "\(word): \(start) - \(end)"
            }

            public var debugDescription: String {
                return "Word(word: \"\(word)\", start: \(start), end: \(end), probability: \(probability), entropy: \(entropy))"
            }

        }

        public var description: String {
            var result = "Segment (\(start) - \(end)):\n"
            for word in words {
                result += "  \(word.description)\n"
            }
            return result
        }

        public var debugDescription: String {
            var result = "Segment(start: \(start), end: \(end), seek: \(seek), tokens: \(tokens.count), words: \(words.count))\n"
            result += "Words:\n"
            for word in words {
                result += "  \(word.debugDescription)\n"
            }
            return result
        }
        
        public init(start: Float, end: Float, tokens: [Rank], words: [Word], seek: Int) {
            self.start = start
            self.end = end
            self.tokens = tokens
            self.words = words
            self.seek = seek
        }
    }

    public func addWordTimestamps(
        segments: inout [Segment],
        alignment: inout [WordTiming],
        prependPunctuations: String = "\"'“¿([{-",
        appendPunctuations: String = "\"'.。,，!！?？:：”)]}、",
        lastSpeechTimestamp: Float = 0.0,
        minDuration: Float = 0.02
    ) {
        guard !segments.isEmpty else { return }

        let wordDurations = alignment.map { $0.end - $0.start }
            .filter { $0 > 0 }

        let medianDuration: Float
        if let median = wordDurations.sorted().middle {
            medianDuration = min(0.7, median)
        } else {
            medianDuration = 0.0
        }
        let maxDuration = medianDuration * 2

        let sentenceEndMarks = Set(".。!！?？")
        for i in 1..<alignment.count {
            let wordDuration = alignment[i].end - alignment[i].start
            if wordDuration > maxDuration {
                if sentenceEndMarks.compatibleContains(alignment[i].word) {
                    alignment[i].end = alignment[i].start + maxDuration
                } else if sentenceEndMarks.compatibleContains(alignment[i - 1].word) {
                    alignment[i].start = alignment[i].end - maxDuration
                }
            }
        }

        mergePunctuations(
            alignment: &alignment,
            prepended: prependPunctuations,
            appended: appendPunctuations
        )

        let timeOffset = Float(segments[0].seek) * HOP_LENGTH / SAMPLE_RATE
        var wordIndex = 0
        var lastSpeechTimestamp = lastSpeechTimestamp

        for var segment in segments {
            var savedTokens = 0
            var words: [Segment.Word] = []

            while wordIndex < alignment.count && savedTokens < segment.tokens.count {
                let timing = alignment[wordIndex]
                let wordStart = timeOffset + timing.start
                var wordEnd = timeOffset + timing.end

                if !timing.word.isEmpty {
                    let wordDuration = wordEnd - wordStart
                    if timing.probability >= 0.6 && wordDuration < minDuration {
                        wordEnd = wordStart + minDuration

                        if wordIndex + 1 < alignment.count {
                            let nextWordStart = timeOffset + alignment[wordIndex + 1].start
                            if wordEnd > nextWordStart {
                                alignment[wordIndex + 1].start = wordEnd - timeOffset
                            }
                        }
                    }

                    words.append(
                        Segment.Word(
                            word: timing.word,
                            start: round(wordStart * 100) / 100,
                            end: round(wordEnd * 100) / 100,
                            probability: timing.probability,
                            entropy: timing.entropy
                        ))
                }

                savedTokens += timing.tokens.count
                wordIndex += 1
            }

            if !words.isEmpty {
                if words[0].end - lastSpeechTimestamp > medianDuration * 4
                    && (words[0].end - words[0].start > maxDuration || (words.count > 1 && words[1].end - words[0].start > maxDuration * 2))
                {

                    if words.count > 1 && words[1].end - words[1].start > maxDuration {
                        let boundary = max(words[1].end / 2, words[1].end - maxDuration)
                        words[0].end = boundary
                        words[1].start = boundary
                    }
                    words[0].start = max(0, words[0].end - maxDuration)
                }

                if segment.start < words[0].end && segment.start - 0.5 > words[0].start {
                    words[0].start = max(0, min(words[0].end - medianDuration, segment.start))
                } else {
                    segment.start = words[0].start
                }

                if segment.end > words[words.count - 1].start && segment.end + 0.5 < words[words.count - 1].end {
                    words[words.count - 1].end = max(
                        words[words.count - 1].start + medianDuration,
                        segment.end
                    )
                } else {
                    segment.end = words[words.count - 1].end
                }

                lastSpeechTimestamp = segment.end
            }
            segment.words = words
        }
    }

    public func mergePunctuations(
        alignment: inout [WordTiming],
        prepended: String,
        appended: String
    ) {
        let prependSet = Set(prepended.map(String.init))
        let appendSet = Set(appended.map(String.init))

        var i = alignment.count - 2
        var j = alignment.count - 1

        while i >= 0 {
            let previous = alignment[i]
            var following = alignment[j]

            if previous.word.hasPrefix(" ") && prependSet.contains(previous.word.trimmingCharacters(in: .whitespaces)) {
                following.word = previous.word + following.word
                following.tokens = previous.tokens + following.tokens
                alignment[j] = following
                alignment[i].word = ""
                alignment[i].tokens = []
            } else {
                j = i
            }
            i -= 1
        }

        i = 0
        j = 1

        while j < alignment.count {
            var previous = alignment[i]
            let following = alignment[j]

            if !previous.word.hasSuffix(" ") && appendSet.contains(following.word) {
                previous.word = previous.word + following.word
                previous.tokens = previous.tokens + following.tokens
                alignment[i] = previous

                alignment[j].word = ""
                alignment[j].tokens = []
            } else {
                i = j
            }

            j += 1
        }
    }
}

extension Array where Element: Comparable {
    var middle: Element? {
        guard !isEmpty else { return nil }
        return self[count / 2]
    }
}


extension WhisperTokenizer {
    struct AlignmentResult {
        let words: [String]
        let wordTokens: [[Rank]]
        let wordBoundaries: [Int]
        let startTimes: [Float]
        let endTimes: [Float]
        let probabilities: [Float]
    }
    


    func findWordAlignments(
        textTokens: [Rank],
        text_indices: [Int],
        time_indices: [Int],
        tokenProbabilities: [Float],  // Token probabilities from Whisper
        tokenEntropies: [Float]  // Token entropies from Whisper
    ) throws -> [WordTiming] {
        var tokensWithEot = textTokens
//        if let eotToken = specialTokens["<|endoftext|>"] {
//            tokensWithEot.append(eotToken)
////            tokensWithEot.insert(specialTokens["<|notimestamps|>"]!, at: 0)
//        }
        let (words, wordTokens) = splitToWordTokens(tokensWithEot)
        guard wordTokens.count > 1 else {
            return []
        }

        var wordBoundaries = [0]
        var cumsum = 0
        for tokens in wordTokens.dropLast() {
            cumsum += tokens.count
            wordBoundaries.append(cumsum)
        }

        var jumps = [true]
        for i in 1..<text_indices.count {
            jumps.append(text_indices[i] != text_indices[i - 1])
        }

        let jumpTimes = zip(jumps, time_indices)
            .filter { $0.0 }
            .map { Float($0.1) / TOKENS_PER_SECOND }

        // Create arrays to store token probabilities and entropies at jump points
        var jumpProbabilities: [Float] = []
        var jumpEntropies: [Float] = []
        var probIndex = 0

        for isJump in jumps {
            if isJump && probIndex < tokenProbabilities.count && probIndex < tokenEntropies.count {
                jumpProbabilities.append(tokenProbabilities[probIndex])
                jumpEntropies.append(tokenEntropies[probIndex])
                probIndex += 1
            }
        }

        var startTimes: [Float] = []
        var endTimes: [Float] = []
        var wordProbabilities: [Float] = []
        var wordEntropies: [Float] = []

        for i in 0..<(wordBoundaries.count - 1) {
            let startIdx = wordBoundaries[i]
            let endIdx = wordBoundaries[i + 1]

            guard startIdx < jumpTimes.count && endIdx < jumpTimes.count else {
                continue
            }

            startTimes.append(jumpTimes[startIdx])
            endTimes.append(jumpTimes[endIdx])

            if startIdx < jumpProbabilities.count && endIdx <= jumpProbabilities.count {
                let tokenProbs = Array(jumpProbabilities[startIdx..<endIdx])
                let wordProb = tokenProbs.isEmpty ? 1.0 : tokenProbs.reduce(0, +) / Float(tokenProbs.count)
                wordProbabilities.append(wordProb)
            } else {
                fatalError("No probability data available")
            }

            if startIdx < jumpEntropies.count && endIdx <= jumpEntropies.count {
                let tokenEnts = Array(jumpEntropies[startIdx..<endIdx])
                let wordEnt = tokenEnts.isEmpty ? 0.0 : tokenEnts.reduce(0, +) / Float(tokenEnts.count)
                wordEntropies.append(wordEnt)
            } else {
                fatalError("No entropy data available")
            }
        }

        var results: [WordTiming] = []
        for i in 0..<min(words.count, startTimes.count, wordProbabilities.count, wordEntropies.count) {
            guard i < endTimes.count && i < wordTokens.count else {
                break
            }

            results.append(
                WordTiming(
                    word: words[i],
                    tokens: wordTokens[i],
                    start: startTimes[i],
                    end: endTimes[i],
                    probability: wordProbabilities[i],
                    entropy: wordEntropies[i]
                ))
        }

        return results
    }
    
    

    public func processSegments(
        segments: inout [Segment],
        tokenizer: WhisperTokenizer,
        numFrames: Int,
        prependPunctuations: String = "\"'“¿([{-",
        appendPunctuations: String = "\"'.。,，!！?？:：”)]}、",
        text_indices: [Int],
        time_indices: [Int],
        probabilities: [Float32],
        tokenEntropies: [Float32],
        pauseThreshold: Float = 0.7
    ) throws {
        guard !segments.isEmpty else { return }

        let textTokensPerSegment = segments.map { segment in
            segment.tokens.filter { $0 < tokenizer.eot }
//            segment.tokens
        }
        let textTokens = Array(textTokensPerSegment.joined())

        var alignment = try findWordAlignments(
            textTokens: textTokens,
            text_indices: text_indices,
            time_indices: time_indices,
            tokenProbabilities: probabilities,
            tokenEntropies: tokenEntropies
        )
        if alignment.isEmpty { return }
//        alignment = Array(alignment[1...])
        let wordDurations = alignment.map { $0.end - $0.start }
            .filter { $0 > 0 }

        guard !wordDurations.isEmpty else { return }

        let medianDuration = min(0.5, Float(wordDurations.sorted()[wordDurations.count / 2]))  // 0.7 vs 0.6 etc.
        let maxDuration = medianDuration * 2.3

        let sentenceEndMarks: [String] = ".。!！?？".map { String($0) }
        for i in 1..<alignment.count {
            if alignment[i].end - alignment[i].start > maxDuration {
                if sentenceEndMarks.contains(alignment[i].word) {
                    alignment[i].end = alignment[i].start + maxDuration
                } else if sentenceEndMarks.contains(alignment[i - 1].word) {
                    alignment[i].start = alignment[i].end - maxDuration
                }
            }
        }

        mergePunctuations(
            alignment: &alignment,
            prepended: prependPunctuations,
            appended: appendPunctuations
        )

        var wordIndex = 0
        var lastSpeechTimestamp: Float = 0

        for (segmentIndex, textTokens) in textTokensPerSegment.enumerated() {
            var savedTokens = 0
            var words: [Segment.Word] = []

            while wordIndex < alignment.count && savedTokens < textTokens.count {
                let timing = alignment[wordIndex]
                if !timing.word.isEmpty {
                    words.append(
                        Segment.Word(
                            word: timing.word,
                            start: timing.start,
                            end:  timing.end,
                            probability: timing.probability,
                            entropy: timing.entropy
                        )
                    )
                }
                savedTokens += timing.tokens.count
                wordIndex += 1
            }

            if !words.isEmpty {
                lastSpeechTimestamp = segments[segmentIndex].end
            }

            segments[segmentIndex].words = words
        }
    }

    private func adjustWordsAfterPause(
        words: inout [Segment.Word],
        fromIndex: Int,
        maxDuration: Float
    ) {
        guard fromIndex < words.count else { return }

        let minDuration: Float = 0.08
        let i = fromIndex
        let duration = words[i].end - words[i].start
        if duration <= minDuration { return }
        if duration >= 2.5 { return } // Fix in restart this is most likely broken word
        if duration > maxDuration * 2 {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration * 1.5,
                words[i].start + duration * 0.7
            )
        }
        else if duration > maxDuration * 1.5 {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration,
                words[i].start + duration * 0.6
            )
        }
        else if duration > maxDuration {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration * 0.9,
                words[i].start + duration * 0.55
            )
        }
        else {
            words[i].start = max(
                words[i].start,
                words[i].start + duration * 0.5
            )
        }
    }
    
    private func adjustWordsAfterPauseLong(
        words: inout [Segment.Word],
        fromIndex: Int,
        maxDuration: Float
    ) {
        guard fromIndex < words.count else { return }

        let minDuration: Float = 0.08
        let i = fromIndex
        let duration = words[i].end - words[i].start
        if duration <= minDuration { return }
        if duration >= 2.5 { return } // Fix in restart this is most likely broken word
        if duration > maxDuration * 2 {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration * 1.25,
                words[i].start + duration * 0.8
            )
        }
        else if duration > maxDuration * 1.5 {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration,
                words[i].start + duration * 0.75
            )
        }
        else if duration > maxDuration {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration * 0.9,
                words[i].start + duration * 0.7
            )
        }
        else {
            words[i].start = max(
                words[i].start,
                words[i].start + duration * 0.65
            )
        }
    }


    
    private func adjustLongWord(
        words: inout [Segment.Word],
        fromIndex: Int,
        maxDuration: Float
    ) {
        guard fromIndex < words.count else { return }

        let minDuration: Float = 0.08
        let i = fromIndex
        let duration = words[i].end - words[i].start
        if duration <= minDuration { return }
        if duration >= 2.5 { return } // Fix in restart this is most likely broken word
        if duration > maxDuration * 2 {
            words[i].start = max(
                words[i].start,
                words[i].end - maxDuration,
                words[i].start + duration * 0.5
            )
        }
        else {
            words[i].start = max(
                words[i].start,
                words[i].start + duration * 0.25
            )
        }
    }


    private func adjustWordTimingsAfterPause(
        words: inout [Segment.Word],
        maxDuration: Float,
        medianDuration: Float,
        pauseThreshold: Float
    ) {
        guard words.count > 1 else { return }
        var i = 1

        while i < words.count {
            let currentWord = words[i]
            let prevWord = words[i - 1]
            

            let gap = currentWord.start - prevWord.end
            let isPause = gap >= pauseThreshold
            let isTooLong = (currentWord.end - currentWord.start) > maxDuration
            if isPause {
                print ("pause, \(currentWord.start), \(currentWord.end), \(currentWord.word)")
                if gap >= 1 {
                    adjustWordsAfterPauseLong(words: &words,
                                              fromIndex: i,
                                              maxDuration: maxDuration)
                }
                else {
                    adjustWordsAfterPause(
                        words: &words,
                        fromIndex: i,
                        maxDuration: maxDuration
                    )
                }


            }
            else if isTooLong {
                print ("too long, \(currentWord.start), \(currentWord.end), \(currentWord.word)")
                adjustLongWord(
                    words: &words,
                    fromIndex: i,
                    maxDuration: maxDuration
                )
            }
            i += 1

        }
    }

    
    private func adjustSegmentBoundaries(
        segment: inout Segment,
        words: inout [Segment.Word],
        medianDuration: Float
    ) {
        if !words.isEmpty {
            segment.start = words[0].start

            if let lastWord = words.last {
                segment.end = lastWord.end
            }
        }
    }

    public struct SegmentResult {
        var tokens: [Int]
        var noSpeechProb: Float
        var avgLogprob: Float
    }

    func processAudioSegments(
        tokens: [Rank],
        tokenizer: WhisperTokenizer,
        result: SegmentResult,
        timeOffset: Float,
        seek: inout Int,
        segmentSize: Int,
        noSpeechThreshold: Float?,
        logprobThreshold: Float?,
        timePrecision: Float
    ) -> [Segment] {
        // Check for no speech
        if let noSpeechThreshold = noSpeechThreshold {
            var shouldSkip = result.noSpeechProb > noSpeechThreshold

            if let logprobThreshold = logprobThreshold,
                result.avgLogprob > logprobThreshold
            {
                shouldSkip = false
            }

            if shouldSkip {
                seek += segmentSize
                return []
            }
        }

        var currentSegments: [Segment] = []

        // Helper functions
        func wordAnomalyScore(_ word: Segment.Word) -> Float {
            let duration = word.end - word.start
            var score: Float = 0.0

            if word.probability < 0.15 {
                score += 1.0
            }
            if duration < 0.133 {
                score += (0.133 - duration) * 15
            }
            if duration > 2.0 {
                score += duration - 2.0
            }
            return score
        }

        func isSegmentAnomaly(_ segment: Segment?) -> Bool {
            guard let segment = segment, !segment.words.isEmpty else { return false }

            let punctuation = Set<String>([",", ".", "!", "?", ";", ":", "'", "\""])
            let words = segment.words.filter { !punctuation.contains($0.word) }.prefix(8)
            let score = words.reduce(0.0) { $0 + wordAnomalyScore($1) }
            return score >= 3 || score + 0.01 >= Float(words.count)
        }

        // Process timestamp tokens
        let timestampTokens = tokens.map { $0 >= tokenizer.timestampBegin }
        let singleTimestampEnding = timestampTokens.suffix(2) == [false, true]

        // Find consecutive timestamps
        var consecutive: [Int] = []
        for i in 0..<(timestampTokens.count - 1) {
            if timestampTokens[i] && timestampTokens[i + 1] {
                consecutive.append(i + 1)
            }
        }

        if !consecutive.isEmpty {
            var slices = consecutive
            if singleTimestampEnding {
                slices.append(tokens.count)
            }

            var lastSlice = 0
            for currentSlice in slices {
                let slicedTokens = Array(tokens[lastSlice..<currentSlice])
                let startTimestampPos = slicedTokens[0] - tokenizer.timestampBegin
                let endTimestampPos = slicedTokens.last! - tokenizer.timestampBegin

                currentSegments.append(
                    Segment(
                        start: timeOffset + Float(startTimestampPos) * timePrecision,
                        end: timeOffset + Float(endTimestampPos) * timePrecision,
                        tokens: Array(slicedTokens),
                        words: [],
                        seek: seek
                    ))

                lastSlice = currentSlice
            }

            if singleTimestampEnding {
                seek += segmentSize
            } else {
                let lastTimestampPos = tokens[lastSlice - 1] - tokenizer.timestampBegin
                seek += Int(lastTimestampPos) * Int(HOP_LENGTH * 2)
            }
        } else {
            var duration = Float(segmentSize) * timePrecision
            let timestamps = tokens.enumerated().filter { timestampTokens[$0.offset] }.map { $0.element }

            if !timestamps.isEmpty && timestamps.last! != tokenizer.timestampBegin {
                let lastTimestampPos = timestamps.last! - tokenizer.timestampBegin
                duration = Float(lastTimestampPos) * timePrecision
            }

            currentSegments.append(
                Segment(
                    start: timeOffset,
                    end: timeOffset + duration,
                    tokens: tokens,
                    words: [],
                    seek: seek
                ))

            seek += segmentSize
        }

        return currentSegments
    }

}

extension WhisperTokenizer {
    public struct TranscriptionSegment {
        let id: Int
        let seek: Int
        var start: Float
        var end: Float
        let text: String
        let tokens: [Rank]
        let temperature: Float
        let avgLogprob: Float
        let compressionRatio: Float
        let noSpeechProb: Float
        var words: [Segment.Word]
    }

}

extension WhisperTokenizer {

    public func createSegmentsFromTokens(
        tokens: [UInt32],
        tokenizer: WhisperTokenizer,
        timePrecision: Float = 0.02
    ) -> [Segment] {
        var currentSegments: [Segment] = []
        let seek = 0
        let timeOffset: Float = 0

        let timestampTokens = tokens.map { $0 >= tokenizer.timestampBegin }
        let singleTimestampEnding = Array(timestampTokens.suffix(2)) == [false, true]

        var consecutive: [Int] = []
        for i in 0..<(timestampTokens.count - 1) {
            if timestampTokens[i] && timestampTokens[i + 1] {
                consecutive.append(i + 1)
            }
        }

        if !consecutive.isEmpty {
            var slices = consecutive
            if singleTimestampEnding {
                slices.append(tokens.count)
            }

            var lastSlice = 0
            for currentSlice in slices {
                let slicedTokens = Array(tokens[lastSlice..<currentSlice])
                guard let firstToken = slicedTokens.first,
                    let lastToken = slicedTokens.last
                else { continue }

//                let startTimestampPos = firstToken - tokenizer.timestampBegin
//                let endTimestampPos = lastToken - tokenizer.timestampBegin

                currentSegments.append(
                    Segment(
                        start: 0,
                        end: 0,
                        tokens: slicedTokens,
                        words: [],
                        seek: seek
                    )
                )

                lastSlice = currentSlice
            }
        } else {
            let timestamps = tokens.enumerated()
                .filter { timestampTokens[$0.offset] }
                .map { $0.element }

            let duration: Float
            if !timestamps.isEmpty && timestamps.last! != UInt32(tokenizer.timestampBegin) {
                let lastTimestampPos = timestamps.last! - tokenizer.timestampBegin
                duration = Float(lastTimestampPos) * timePrecision
            } else {
                duration = Float(tokens.count) * timePrecision
            }

            currentSegments.append(
                Segment(
                    start: timeOffset,
                    end: timeOffset + duration,
                    tokens: tokens,
                    words: [],
                    seek: seek
                )
            )
        }

        return currentSegments
    }

}

struct SegmentResult {
    var tokens: [Int]
    var noSpeechProb: Float
    var avgLogprob: Float
}

extension WhisperTokenizer.Segment {
    static func generateSRT(from segments: [WhisperTokenizer.Segment]) -> String {
        var srtOutput = ""
        var index = 1

        for segment in segments {
            srtOutput += "\(index)\n"

            let startTime = formatTime(segment.start)
            let endTime = formatTime(segment.end)
            srtOutput += "\(startTime) --> \(endTime)\n"

            let text = segment.words
                .map { $0.word.trimmingCharacters(in: .whitespaces) }
                .joined(separator: " ")
            srtOutput += "\(text)\n"

            if index < segments.count {
                srtOutput += "\n"
            }

            index += 1
        }
        return srtOutput
    }

    private static func formatTime(_ timeInSeconds: Float) -> String {
        let hours = timeInSeconds / 3600
        let minutes = timeInSeconds.truncatingRemainder(dividingBy: 3600) / 60
        let seconds = timeInSeconds.truncatingRemainder(dividingBy: 60)
        let milliseconds = seconds.truncatingRemainder(dividingBy: 1) * 1000

        return String(format: "%02.0f:%02.0f:%02.0f,%03.0f", hours, minutes, seconds, milliseconds)
    }
}

public func printSegments(_ segments: [WhisperTokenizer.Segment]) {
    for (index, segment) in segments.enumerated() {
        print("Segment \(index) (\(segment.start) - \(segment.end)):")
        for word in segment.words {
            print("  \(word.word): \(word.start) - \(word.end)")
        }
        if index < segments.count - 1 {
            print()
        }
    }
}

public func debugPrintSegments(_ segments: [WhisperTokenizer.Segment]) {
    print("[")
    for (index, segment) in segments.enumerated() {
        print("  \(segment.debugDescription)")
        if index < segments.count - 1 {
            print()
        }
    }
    print("]")
}
