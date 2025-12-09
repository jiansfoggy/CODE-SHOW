import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        
        let cameraVC = CameraViewController()
        let navController = UINavigationController(rootViewController: cameraVC)
        
        window?.rootViewController = navController
        window?.makeKeyAndVisible()
        
        return true
    }
}
