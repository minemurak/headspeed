// Checks the Swift ball lock against the Python reference (reference/balllock.py) on the same
// frames. Built with swiftc together with HeadSpeed/Analysis/*.swift in CI:
//   python3 reference/export_parity.py /tmp/parity
//   swiftc -O HeadSpeed/Analysis/*.swift Tests/Parity/main.swift -o parity && ./parity /tmp/parity
import Foundation

let dir = CommandLine.arguments[1]
let cases = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "/cases.json"))) as! [String: Any]
let frameInfo = cases["frames"] as! [String: [String: Int]]
var frames: [String: GrayImage] = [:]
for (name, info) in frameInfo {
    let data = try! Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name).gray"))
    frames[name] = GrayImage(width: info["width"]!, height: info["height"]!, pixels: [UInt8](data))
}

var failures = 0
func check(_ name: String, _ ok: Bool, _ info: String = "") {
    print(ok ? "OK  " : "FAIL", name, info)
    if !ok { failures += 1 }
}
func near(_ a: Double, _ b: Double, _ tol: Double = 0.05) -> Bool { abs(a - b) <= tol }
func fmt(_ f: BallFix?) -> String {
    guard let f else { return "nil" }
    return String(format: "(%.2f, %.2f, %.2f)", f.x, f.y, f.d)
}
func candidate(_ o: Any?) -> BallCandidate? {
    guard let c = o as? [String: Any] else { return nil }
    return BallCandidate(fix: BallFix(x: c["x"] as! Double, y: c["y"] as! Double, d: c["d"] as! Double),
                         kind: BallKind(rawValue: c["kind"] as! String)!)
}
func same(_ a: BallCandidate, _ b: BallCandidate) -> Bool {
    a.kind == b.kind && near(a.fix.x, b.fix.x) && near(a.fix.y, b.fix.y) && near(a.fix.d, b.fix.d)
}

for (i, c) in (cases["locate"] as! [[String: Any]]).enumerated() {
    let name = c["frame"] as! String
    let anchor = (c["anchor"] as? [Double]).map { ($0[0], $0[1]) }
    let t0 = Date()
    let got = BallLocator.locate(frames[name]!, rear: c["rear"] as! Bool, anchor: anchor,
                                 home: candidate(c["home"]), horizon: c["horizon"] as? Double)
    let ms = Date().timeIntervalSince(t0) * 1000
    let want = (c["result"] as! [Any]).compactMap(candidate)
    let n = min(want.count, 4)
    let ok = got.count >= n && zip(got.prefix(n), want).allSatisfy(same) && (want.count == 4 || got.count == want.count)
    check("locate #\(i) \(name)", ok, String(format: "%.0f ms; ", ms) +
          "swift \(got.prefix(2).map { "\($0.kind.rawValue)\(fmt($0.fix))" }) python \(want.prefix(2).map { "\($0.kind.rawValue)\(fmt($0.fix))" })")
}

for (i, c) in (cases["follow"] as! [[String: Any]]).enumerated() {
    var f = BallFollower(frames[c["start"] as! String]!, candidate(c["cand"])!)
    let names = c["frames"] as! [String]
    var ok = true, info = ""
    for (k, (fr, s)) in zip(names, c["steps"] as! [[String: Any]]).enumerated() {
        let st = f.step(t: Double(k + 1) / 40, frames[fr]!)
        let want = (s["fix"] as? [Double]).map { BallFix(x: $0[0], y: $0[1], d: $0[2]) }
        let state: String, fix: BallFix?
        switch st {
        case .locked(let x): state = "locked"; fix = x
        case .following(let x): state = "following"; fix = x
        case .lost: state = "lost"; fix = nil
        }
        let same = state == s["state"] as! String && (fix == nil) == (want == nil)
            && (fix == nil || (near(fix!.x, want!.x) && near(fix!.y, want!.y) && near(fix!.d, want!.d)))
        if !same && ok { ok = false; info = "step \(k + 1): swift \(state) \(fmt(fix)) python \(s["state"]!) \(fmt(want))" }
    }
    check("follow #\(i) \(c["start"]!)", ok, info)
}

for (i, c) in (cases["lock"] as! [[String: Any]]).enumerated() {
    var lock = BallLock()
    var ok = true
    for (k, (s, r)) in zip(c["seq"] as! [Any], c["result"] as! [Any]).enumerated() {
        let v = s as? [Double]
        let got = lock.update(t: Double(k) / 40, fix: v.map { BallFix(x: $0[0], y: $0[1], d: $0[2]) }, change: v?[3] ?? 0)
        let want = (r as? [Double]).map { BallFix(x: $0[0], y: $0[1], d: $0[2]) }
        if (got == nil) != (want == nil) || (got != nil && !(near(got!.x, want!.x) && near(got!.y, want!.y) && near(got!.d, want!.d))) { ok = false }
    }
    check("lock #\(i)", ok)
}

print(failures == 0 ? "OK" : "\(failures) failed")
exit(failures == 0 ? 0 : 1)
