import Foundation

enum Fmt {
    static func num(_ v: Double?, _ digits: Int) -> String {
        guard let v else { return "—" }
        return String(format: "%.\(digits)f", v)
    }

    static func signed(_ v: Double?, _ digits: Int) -> String {
        guard let v else { return "—" }
        return String(format: "%+.\(digits)f", v)
    }
}
