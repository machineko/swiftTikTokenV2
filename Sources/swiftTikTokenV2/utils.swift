//import AVFoundation
import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

#if arch(x86_64)
    public typealias Float16 = UInt16
#endif

public struct ComplexHalf {

    var real: Float16
    var imag: Float16
    var mag: Float16

    static func size() -> Int {
        return MemoryLayout<ComplexHalf>.stride
    }
}

public struct AudioParams {
    var frameSize: UInt32
    var hopLength: UInt32
    var numFrames: UInt32
    var melFilters: UInt32
    var sampleRate: Float

    public init(
        frameSize: UInt32 = 400,
        hopLength: UInt32 = 160,
        numFrames: UInt32 = 2998,
        melFilters: UInt32 = 128,
        sampleRate: Float32 = 16000
    ) {
        self.frameSize = frameSize
        self.hopLength = hopLength
        self.numFrames = numFrames
        self.melFilters = melFilters
        self.sampleRate = sampleRate
    }

    public static func size() -> Int {
        return MemoryLayout<Self>.size
    }
}

func initializeAudioParameters() -> AudioParams {
    let frameSize: UInt32 = 400
    let hopLength: UInt32 = 160
    let numFrames: UInt32 = (30 * 16000 - 400) / 160 + 1

    return AudioParams(
        frameSize: frameSize,
        hopLength: hopLength,
        numFrames: numFrames,
        melFilters: 128,
        sampleRate: 16000.0
    )
}

public struct DTWResult {
    let path: [(Int, Int)]
    let cost: Float
}

public enum DTWError: Error {
    case initializationFailed
    case invalidDimensions
    case encoderCreationFailed
}

public struct DTW {
    public init() {

    }
    func backtrace(trace: [Float], N: Int, M: Int) -> (text_indices: [Int], time_indices: [Int]) {
        var i = N
        var j = M

        var mutableTrace = trace
        for idx in 0...M {
            mutableTrace[idx] = 2
        }
        for idx in 0...N {
            mutableTrace[idx * (M + 1)] = 1
        }

        var textIndices: [Int] = []
        var timeIndices: [Int] = []

        while i > 0 || j > 0 {
            let traceValue = mutableTrace[i * (M + 1) + j]

            textIndices.append(i - 1)
            timeIndices.append(j - 1)

            switch traceValue {
            case 0:
                i -= 1
                j -= 1
            case 1:
                i -= 1
            case 2:
                j -= 1
            default:
                fatalError("Unexpected trace value: \(traceValue)")
            }
        }

        return (textIndices.reversed(), timeIndices.reversed())
    }

    public func dtw(input: MTLBuffer, N: Int, M: Int) -> (text_indices: [Int], time_indices: [Int]) {
        let inputPtr = input.contents().bindMemory(to: Float32.self, capacity: input.length / MemoryLayout<Float32>.stride)

        var cost = [Float32](repeating: Float32.infinity, count: (N + 1) * (M + 1))
        var trace = [Float32](repeating: -1, count: (N + 1) * (M + 1))

        cost[0] = 0

        for j in 1...M {
            for i in 1...N {
                let c0 = cost[(i - 1) * (M + 1) + (j - 1)]  // diagonal
                let c1 = cost[(i - 1) * (M + 1) + j]  // vertical
                let c2 = cost[i * (M + 1) + (j - 1)]  // horizontal

                let currentCost = inputPtr[(i - 1) * M + (j - 1)]

                let (c, t): (Float32, Float32)
                if c0 <= c1 && c0 <= c2 {
                    c = c0
                    t = 0
                } else if c1 <= c2 {
                    c = c1
                    t = 1
                } else {
                    c = c2
                    t = 2
                }

                cost[i * (M + 1) + j] = currentCost + c
                trace[i * (M + 1) + j] = t
            }
        }

        return backtrace(trace: trace, N: N, M: M)
    }

}

struct AttentionMaskKernel {
    private let pipelineState: MTLComputePipelineState

    init?(device: MTLDevice) throws {
        guard let library = try? device.makeDefaultLibrary(bundle: Bundle.main) else {
            throw DTWError.initializationFailed
        }
        guard let kernelFunction = library.makeFunction(name: "createPaddingAttentionMask"),
            let pipelineState = try? device.makeComputePipelineState(function: kernelFunction)
        else {
            throw DTWError.initializationFailed
        }
        self.pipelineState = pipelineState
    }

    func encodeKernel(
        commandBuffer: MPSCommandBuffer,
        outputBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        maxLength: Int
    ) {

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (maxLength + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (maxLength + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)

        encoder.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
    }
}

extension ComplexHalf {
    static func makeBuffer(device: MTLDevice, count: Int) -> MTLBuffer? {
        return device.makeBuffer(length: ComplexHalf.size() * count, options: .storageModeShared)
    }

    static func makeBuffer(device: MTLDevice, array: [ComplexHalf]) -> MTLBuffer? {
        return device.makeBuffer(bytes: array, length: ComplexHalf.size() * array.count, options: .storageModeShared)
    }
}

extension AudioParams {
    func toMetalBuffer(device: MTLDevice) -> MTLBuffer? {
        var params = self
        return device.makeBuffer(bytes: &params, length: AudioParams.size(), options: .storageModeShared)
    }
}

extension MTLFunctionConstantValues {
    func setConstant(_ value: inout UInt32, at index: Int) {
        setConstantValue(&value, type: .uint, index: index)
    }

    func setConstant(_ value: inout Bool, at index: Int) {
        setConstantValue(&value, type: .bool, index: index)
    }
}

func reverseBits(value: UInt, bitCount: UInt) -> UInt {
    var reversedValue: UInt = 0
    for i in 0..<bitCount {
        if (value & (1 << i)) != 0 {
            reversedValue |= 1 << (bitCount - 1 - i)
        }
    }
    return reversedValue
}

struct AudioSegment {
    var samples: [Float32]
    let sampleRate: Double
    let isLastSegment: Bool
}

func trimPadAudio(samples: [Float32], sampleRate: Double = 16000) -> [AudioSegment] {
    let targetDuration: Double = 30.0
    let samplesFor30Seconds = Int(targetDuration * sampleRate)
    var segments: [AudioSegment] = []

    if samples.count < samplesFor30Seconds {
        var paddedSamples = samples
        paddedSamples.append(contentsOf: [Float32](repeating: 0.0, count: samplesFor30Seconds - samples.count))
        segments.append(AudioSegment(samples: paddedSamples, sampleRate: sampleRate, isLastSegment: true))
    } else {
        let numberOfFullSegments = (samples.count + samplesFor30Seconds - 1) / samplesFor30Seconds

        for i in 0..<numberOfFullSegments {
            let start = i * samplesFor30Seconds
            let end = min(start + samplesFor30Seconds, samples.count)
            var segment = Array(samples[start..<end])

            if end - start < samplesFor30Seconds {
                segment.append(contentsOf: [Float32](repeating: 0.0, count: samplesFor30Seconds - (end - start)))
            }

            segments.append(
                AudioSegment(
                    samples: segment,
                    sampleRate: sampleRate,
                    isLastSegment: i == numberOfFullSegments - 1
                ))
        }
    }

    return segments
}
