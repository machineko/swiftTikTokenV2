// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swiftTikTokenV2",
    platforms: [.macOS(.v12), .iOS(.v14), .tvOS(.v14)],

    products: [
        .library(
            name: "swiftTikTokenV2",
            targets: ["swiftTikTokenV2"]),
    ],
    targets: [
        .target(
            name: "swiftTikTokenV2",
            resources: [
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "swiftTikTokenV2Tests",
            dependencies: ["swiftTikTokenV2"]
        ),
    ]
)
