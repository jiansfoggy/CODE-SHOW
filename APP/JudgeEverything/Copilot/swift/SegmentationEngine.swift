import UIKit
import Vision
import CoreML
import AVFoundation

/// Wrapper class to load and run YOLOv9 and MobileSAM Core ML models
class SegmentationEngine {
    var yoloModel: MLModel?
    var samModel: MLModel?
    
    init() {
        loadModels()
    }
    
    func loadModels() {
        // Load YOLOv9 model
        if let yoloURL = Bundle.main.url(forResource: "yolov9-c", withExtension: "mlmodel") {
            do {
                let compiled = try MLModel.compileModel(at: yoloURL)
                yoloModel = try MLModel(contentsOf: compiled)
                print("YOLOv9 loaded successfully")
            } catch {
                print("Failed to load YOLOv9: \(error)")
            }
        }
        
        // Load MobileSAM model
        if let samURL = Bundle.main.url(forResource: "mobile_sam", withExtension: "mlmodel") {
            do {
                let compiled = try MLModel.compileModel(at: samURL)
                samModel = try MLModel(contentsOf: compiled)
                print("MobileSAM loaded successfully")
            } catch {
                print("Failed to load MobileSAM: \(error)")
            }
        }
    }
    
    /// Run YOLOv9 detection on an image
    func detectObjects(in image: UIImage) -> [(bbox: CGRect, confidence: Float, classId: Int)]? {
        guard let model = yoloModel else { return nil }
        guard let pixelBuffer = image.toCVPixelBuffer() else { return nil }
        
        do {
            // Prepare input for YOLOv9
            let input = yolov9CInput(input_image: pixelBuffer)
            let output = try model.prediction(from: input)
            
            // Parse detection outputs (adjust based on actual model output structure)
            // This is a placeholder - actual parsing depends on model's output format
            var detections: [(bbox: CGRect, confidence: Float, classId: Int)] = []
            
            // Example parsing (adjust to match actual YOLOv9 output structure)
            if let outputFeatures = output.featureProvider as? MLDictionaryFeatureProvider {
                for key in outputFeatures.featureNames {
                    print("Output key: \(key)")
                }
            }
            
            return detections
        } catch {
            print("Detection error: \(error)")
            return nil
        }
    }
    
    /// Run MobileSAM mask generation on an image using a detection box
    func generateMask(for image: UIImage, in box: CGRect) -> UIImage? {
        guard let model = samModel else { return nil }
        guard let pixelBuffer = image.toCVPixelBuffer() else { return nil }
        
        do {
            // Prepare input for MobileSAM
            let input = MobileSAMInput(input_image: pixelBuffer)
            let output = try model.prediction(from: input)
            
            // Extract mask from output and convert to UIImage
            // Placeholder - adjust based on actual MobileSAM output format
            if let maskBuffer = output.featureValue(for: "output")?.multiArrayValue {
                return maskFromMLMultiArray(maskBuffer)
            }
            
            return nil
        } catch {
            print("Mask generation error: \(error)")
            return nil
        }
    }
    
    /// Helper: convert MLMultiArray to UIImage
    private func maskFromMLMultiArray(_ multiArray: MLMultiArray) -> UIImage? {
        guard multiArray.shape.count >= 2 else { return nil }
        
        let height = Int(multiArray.shape[0].intValue)
        let width = Int(multiArray.shape[1].intValue)
        
        var pixelData = [UInt8](repeating: 0, count: height * width)
        for i in 0..<(height * width) {
            let value = multiArray[NSNumber(value: i)].floatValue
            pixelData[i] = UInt8(value > 0.5 ? 255 : 0)
        }
        
        if let cgImage = createCGImageFromGrayscale(pixelData, width: width, height: height) {
            return UIImage(cgImage: cgImage)
        }
        return nil
    }
    
    private func createCGImageFromGrayscale(_ data: [UInt8], width: Int, height: Int) -> CGImage? {
        let bitsPerComponent = 8
        let bytesPerPixel = 1
        let bytesPerRow = bytesPerPixel * width
        
        let dataProvider = CGDataProvider(data: NSData(bytes: data, length: height * bytesPerRow))
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        return CGImage(width: width,
                       height: height,
                       bitsPerComponent: bitsPerComponent,
                       bitsPerPixel: bitsPerComponent * bytesPerPixel,
                       bytesPerRow: bytesPerRow,
                       space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: dataProvider!,
                       decode: nil,
                       shouldInterpolate: false,
                       intent: .defaultIntent)
    }
}

// MARK: - UIImage Extensions

extension UIImage {
    /// Convert UIImage to CVPixelBuffer for Core ML input
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(self.size.width),
            Int(self.size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess else { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: pixelData,
            width: Int(self.size.width),
            height: Int(self.size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
}

// MARK: - Model Input/Output Classes (Auto-generated by Xcode)
// Replace these with actual generated classes from your .mlmodel files

class yolov9CInput {
    var input_image: CVPixelBuffer
    
    init(input_image: CVPixelBuffer) {
        self.input_image = input_image
    }
}

class MobileSAMInput {
    var input_image: CVPixelBuffer
    
    init(input_image: CVPixelBuffer) {
        self.input_image = input_image
    }
}
