import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("ERROR: \(message)\n").utf8))
    exit(1)
}

func frameStatistics(_ image: CGImage) -> (mean: Double, standardDeviation: Double, range: Int, brightFraction: Double) {
    let width = 96
    let height = 54
    let bytesPerRow = width * 4
    var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
    pixels.withUnsafeMutableBytes { rawBuffer in
        guard let context = CGContext(
            data: rawBuffer.baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            fail("could not create frame-analysis context")
        }
        context.interpolationQuality = .low
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }

    var values = [Double]()
    values.reserveCapacity(width * height)
    var minimum = 255
    var maximum = 0
    var bright = 0
    for offset in stride(from: 0, to: pixels.count, by: 4) {
        let red = Int(pixels[offset])
        let green = Int(pixels[offset + 1])
        let blue = Int(pixels[offset + 2])
        let luma = (54 * red + 183 * green + 19 * blue) / 256
        minimum = min(minimum, luma)
        maximum = max(maximum, luma)
        if luma >= 24 { bright += 1 }
        values.append(Double(luma))
    }
    let mean = values.reduce(0, +) / Double(values.count)
    let variance = values.reduce(0) { partial, value in
        let delta = value - mean
        return partial + delta * delta
    } / Double(values.count)
    return (
        mean,
        sqrt(variance),
        maximum - minimum,
        Double(bright) / Double(values.count)
    )
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: audit_local_video_frames.swift VIDEO_MP4")
}

let videoURL = URL(fileURLWithPath: CommandLine.arguments[1])
let asset = AVURLAsset(url: videoURL)
let duration = CMTimeGetSeconds(asset.duration)
guard duration.isFinite, duration > 0 else { fail("invalid video duration") }
let videoTracks = asset.tracks(withMediaType: .video)
let audioTracks = asset.tracks(withMediaType: .audio)
guard videoTracks.count == 1 else {
    fail("expected exactly one video track")
}
guard audioTracks.count == 1 else {
    fail("expected exactly one audio track")
}

var sampleSeconds = stride(from: 0.5, to: duration, by: 2.0).map { $0 }
for boundary in [30.0, 58.0, 104.0, 137.0, 163.0] {
    sampleSeconds.append(boundary - 0.20)
    sampleSeconds.append(boundary + 0.20)
}
sampleSeconds.append(max(0.0, duration - 0.25))
sampleSeconds = Array(Set(sampleSeconds.map { Int(($0 * 100).rounded()) }))
    .map { Double($0) / 100.0 }
    .sorted()

let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
var weakestStandardDeviation = Double.greatestFiniteMagnitude
var weakestRange = Int.max
var weakestBrightFraction = Double.greatestFiniteMagnitude
var failures = [String]()
var observedWidth: Int?
var observedHeight: Int?

for second in sampleSeconds {
    var actual = CMTime.zero
    let requested = CMTime(seconds: second, preferredTimescale: 600)
    let image: CGImage
    do {
        image = try generator.copyCGImage(at: requested, actualTime: &actual)
    } catch {
        fail("could not extract frame at \(second)s: \(error)")
    }
    if observedWidth == nil {
        observedWidth = image.width
        observedHeight = image.height
    } else if observedWidth != image.width || observedHeight != image.height {
        fail("video frame dimensions changed during playback")
    }
    let statistics = frameStatistics(image)
    weakestStandardDeviation = min(weakestStandardDeviation, statistics.standardDeviation)
    weakestRange = min(weakestRange, statistics.range)
    weakestBrightFraction = min(weakestBrightFraction, statistics.brightFraction)
    if statistics.standardDeviation < 2.0 || statistics.range < 20 || statistics.brightFraction < 0.002 {
        failures.append(
            String(
                format: "%.2fs(mean=%.3f,std=%.3f,range=%d,bright=%.5f)",
                second,
                statistics.mean,
                statistics.standardDeviation,
                statistics.range,
                statistics.brightFraction
            )
        )
    }
}

if !failures.isEmpty {
    fail("blank-or-solid frame samples: \(failures.joined(separator: ", "))")
}
guard observedWidth == 1920, observedHeight == 1080 else {
    fail("expected 1920x1080 video frames")
}

print("video_frame_samples=\(sampleSeconds.count)")
print(String(format: "video_duration_seconds=%.3f", duration))
print("video_tracks=\(videoTracks.count)")
print("audio_tracks=\(audioTracks.count)")
print("video_dimensions=\(observedWidth!)x\(observedHeight!)")
print(String(format: "minimum_frame_standard_deviation=%.3f", weakestStandardDeviation))
print("minimum_frame_luma_range=\(weakestRange)")
print(String(format: "minimum_frame_bright_fraction=%.5f", weakestBrightFraction))
print("blank_or_solid_frame_samples=0")
