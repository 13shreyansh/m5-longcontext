import AppKit
import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

let width = 1920
let height = 1080
let timescale: CMTimeScale = 600

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("ERROR: \(message)\n").utf8))
    exit(1)
}

func pixelBuffer(for imageURL: URL) -> CVPixelBuffer {
    guard let image = NSImage(contentsOf: imageURL) else {
        fail("could not load \(imageURL.path)")
    }
    var proposed = NSRect(x: 0, y: 0, width: width, height: height)
    guard let cgImage = image.cgImage(forProposedRect: &proposed, context: nil, hints: nil) else {
        fail("could not rasterize \(imageURL.path)")
    }
    let attributes: CFDictionary = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
    ] as CFDictionary
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        attributes,
        &buffer
    )
    guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
        fail("CVPixelBufferCreate failed with \(status)")
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard let context = CGContext(
        data: CVPixelBufferGetBaseAddress(pixelBuffer),
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue
    ) else {
        fail("could not create bitmap context")
    }
    context.setFillColor(NSColor.black.cgColor)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    return pixelBuffer
}

func waitUntilReady(_ input: AVAssetWriterInput) {
    while !input.isReadyForMoreMediaData {
        Thread.sleep(forTimeInterval: 0.01)
    }
}

guard CommandLine.arguments.count == 4 else {
    fail("usage: build_local_video_draft.swift OUTPUT_DIR AUDIO_AIFF DURATION_SECONDS")
}

let outputDirectory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let audioURL = URL(fileURLWithPath: CommandLine.arguments[2])
guard let durationSeconds = Int(CommandLine.arguments[3]), durationSeconds > 0,
      durationSeconds <= 180 else {
    fail("duration must be between 1 and 180 seconds")
}
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
let assets = root.appendingPathComponent("docs/video_assets", isDirectory: true)
let hook = assets.appendingPathComponent("00_hook.svg")
let architecture = assets.appendingPathComponent("01_architecture.svg")
let results = assets.appendingPathComponent("02_results.svg")
let boundaries = assets.appendingPathComponent("03_boundaries.svg")
let row14Evidence = assets.appendingPathComponent("04_row14_evidence.svg")
let reproducibility = assets.appendingPathComponent("05_reproducibility.svg")
let silentURL = outputDirectory.appendingPathComponent("silent_cards.mp4")
let finalURL = outputDirectory.appendingPathComponent("track3_local_draft.mp4")

let writer: AVAssetWriter
do {
    writer = try AVAssetWriter(outputURL: silentURL, fileType: .mp4)
} catch {
    fail("could not create video writer: \(error)")
}
let videoSettings: [String: Any] = [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: width,
    AVVideoHeightKey: height,
    AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: 4_000_000,
        AVVideoMaxKeyFrameIntervalKey: 1,
    ],
]
let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
videoInput.expectsMediaDataInRealTime = false
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: videoInput,
    sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
    ]
)
guard writer.canAdd(videoInput) else { fail("video writer rejected its input") }
writer.add(videoInput)
guard writer.startWriting() else {
    fail("video writer failed to start: \(String(describing: writer.error))")
}
writer.startSession(atSourceTime: .zero)

let frames: [(Int, URL)] = [
    (0, hook),
    (30, results),
    (58, architecture),
    (104, row14Evidence),
    (137, reproducibility),
    (163, boundaries),
    (durationSeconds + 1, boundaries),
]
for (second, imageURL) in frames {
    waitUntilReady(videoInput)
    let time = CMTime(seconds: Double(second), preferredTimescale: timescale)
    guard adaptor.append(pixelBuffer(for: imageURL), withPresentationTime: time) else {
        fail("failed to append frame at \(second)s: \(String(describing: writer.error))")
    }
}
videoInput.markAsFinished()
let writerSemaphore = DispatchSemaphore(value: 0)
writer.finishWriting { writerSemaphore.signal() }
writerSemaphore.wait()
guard writer.status == .completed else {
    fail("video writer ended with \(writer.status.rawValue): \(String(describing: writer.error))")
}

let duration = CMTime(seconds: Double(durationSeconds), preferredTimescale: timescale)
let composition = AVMutableComposition()
let silentAsset = AVURLAsset(url: silentURL)
guard let sourceVideo = silentAsset.tracks(withMediaType: .video).first,
      let destinationVideo = composition.addMutableTrack(
        withMediaType: .video,
        preferredTrackID: kCMPersistentTrackID_Invalid
      ) else {
    fail("silent draft has no video track")
}
do {
    try destinationVideo.insertTimeRange(
        CMTimeRange(start: .zero, duration: duration),
        of: sourceVideo,
        at: .zero
    )
} catch {
    fail("could not add video track: \(error)")
}
destinationVideo.preferredTransform = sourceVideo.preferredTransform

let audioAsset = AVURLAsset(url: audioURL)
let audioDurationSeconds = CMTimeGetSeconds(audioAsset.duration)
guard audioDurationSeconds.isFinite,
      audioDurationSeconds > 0,
      audioDurationSeconds <= Double(durationSeconds) + 0.1 else {
    fail(
        "narration does not fit the video timeline: audio=\(audioDurationSeconds)s "
        + "timeline=\(durationSeconds)s"
    )
}
if let sourceAudio = audioAsset.tracks(withMediaType: .audio).first,
   let destinationAudio = composition.addMutableTrack(
        withMediaType: .audio,
        preferredTrackID: kCMPersistentTrackID_Invalid
   ) {
    let audioDuration = CMTimeCompare(audioAsset.duration, duration) < 0
        ? audioAsset.duration : duration
    do {
        try destinationAudio.insertTimeRange(
            CMTimeRange(start: .zero, duration: audioDuration),
            of: sourceAudio,
            at: .zero
        )
    } catch {
        fail("could not add narration track: \(error)")
    }
} else {
    fail("narration asset has no audio track")
}

guard let exporter = AVAssetExportSession(
    asset: composition,
    presetName: AVAssetExportPresetHighestQuality
) else {
    fail("could not create final exporter")
}
exporter.outputURL = finalURL
exporter.outputFileType = .mp4
exporter.timeRange = CMTimeRange(start: .zero, duration: duration)
let exportSemaphore = DispatchSemaphore(value: 0)
exporter.exportAsynchronously { exportSemaphore.signal() }
exportSemaphore.wait()
guard exporter.status == .completed else {
    fail("export failed with \(exporter.status.rawValue): \(String(describing: exporter.error))")
}

let finalAsset = AVURLAsset(url: finalURL)
let finalDuration = CMTimeGetSeconds(finalAsset.duration)
let videoTracks = finalAsset.tracks(withMediaType: .video).count
let audioTracks = finalAsset.tracks(withMediaType: .audio).count
guard abs(finalDuration - Double(durationSeconds)) <= 0.1,
      videoTracks == 1,
      audioTracks == 1 else {
    fail(
        "unexpected final media: duration=\(finalDuration) "
        + "video_tracks=\(videoTracks) audio_tracks=\(audioTracks)"
    )
}
let formattedDuration = String(format: "%.3f", finalDuration)
let formattedAudioDuration = String(format: "%.3f", audioDurationSeconds)
print("video_duration_seconds=\(formattedDuration)")
print("narration_duration_seconds=\(formattedAudioDuration)")
print("video_dimensions=1920x1080")
print("video_tracks=\(videoTracks) audio_tracks=\(audioTracks)")
print("publication_performed=false")
