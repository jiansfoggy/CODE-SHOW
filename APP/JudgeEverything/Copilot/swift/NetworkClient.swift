import Foundation
import UIKit

class NetworkClient {
    let serverURL: URL

    init(server: String) {
        self.serverURL = URL(string: server)! // e.g. "http://192.168.1.100:8000/segment"
    }

    func sendFrame(_ image: UIImage, click: CGPoint?, completion: @escaping (Result<[String:Any], Error>) -> Void) {
        guard let url = URL(string: serverURL.absoluteString) else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var data = Data()
        // image as JPEG
        if let jpeg = image.jpegData(compressionQuality: 0.7) {
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\n".data(using: .utf8)!)
            data.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            data.append(jpeg)
            data.append("\r\n".data(using: .utf8)!)
        }

        // clicks as JSON string
        let clicksArr: [[CGFloat]]
        if let c = click {
            clicksArr = [[c.x, c.y]]
        } else {
            clicksArr = []
        }
        if let clicksJson = try? JSONSerialization.data(withJSONObject: clicksArr, options: []) {
            data.append("--\(boundary)\r\n".data(using: .utf8)!)
            data.append("Content-Disposition: form-data; name=\"clicks\"\r\n\r\n".data(using: .utf8)!)
            data.append(clicksJson)
            data.append("\r\n".data(using: .utf8)!)
        }

        data.append("--\(boundary)--\r\n".data(using: .utf8)!)

        let task = URLSession.shared.uploadTask(with: request, from: data) { respData, resp, err in
            if let err = err {
                completion(.failure(err))
                return
            }
            guard let respData = respData else {
                completion(.failure(NSError(domain: "nil", code: -1)))
                return
            }
            do {
                if let json = try JSONSerialization.jsonObject(with: respData, options: []) as? [String:Any] {
                    completion(.success(json))
                } else {
                    completion(.failure(NSError(domain: "json", code: -2)))
                }
            } catch {
                completion(.failure(error))
            }
        }
        task.resume()
    }
}
