import Foundation

/// Golf ball diameter used as the on-screen ruler (mm).
let ballDiameterMM = 42.67

enum Club: String, CaseIterable, Identifiable, Codable {
    case driver = "1W", fairway = "FW", utility = "UT", iron7 = "7I", wedge = "W"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .driver: return "ドライバー"
        case .fairway: return "FW"
        case .utility: return "UT"
        case .iron7: return "7I"
        case .wedge: return "ウェッジ"
        }
    }

    /// Assumed swing radius (shoulder to clubhead, m). Used only to turn the
    /// measured chord of the arc into the tangent at impact (attack angle).
    var swingRadius: Double {
        switch self {
        case .driver: return 1.6
        case .fairway: return 1.55
        case .utility: return 1.5
        case .iron7: return 1.4
        case .wedge: return 1.3
        }
    }

    /// Typical smash factor range for a well-struck shot.
    var smashRange: ClosedRange<Double> {
        switch self {
        case .driver: return 1.44...1.50
        case .fairway: return 1.40...1.47
        case .utility: return 1.37...1.44
        case .iron7: return 1.33...1.38
        case .wedge: return 1.15...1.25
        }
    }
}

struct SwingResult: Codable, Identifiable, Equatable {
    var id = UUID()
    var date = Date()
    var club: Club
    var headSpeed: Double?      // m/s
    var ballSpeed: Double?      // m/s
    var smash: Double?
    var launch: Double?         // deg, up positive
    var attack: Double?         // deg, down negative
    var backswing: Double?      // s
    var downswing: Double?      // s
    var shutter: Double?        // s, exposure duration while recording
    var notes: [String] = []

    var tempo: Double? {
        guard let b = backswing, let d = downswing, d > 0 else { return nil }
        return b / d
    }

    /// Rule of thumb: driver carry (yd) ≈ ball speed (m/s) × 4.
    var carryYards: Double? {
        guard club == .driver, let bs = ballSpeed else { return nil }
        return bs * 4
    }

    var hasMeasurement: Bool { headSpeed != nil || ballSpeed != nil }
}
