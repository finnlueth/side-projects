import Foundation

struct Quote: Codable, Equatable {
    let symbol: String
    let price: Double
    let change: Double
    let changePercent: Double
    let latestTradingDay: String
    let fetchedAt: Date
    /// "$" for stocks and crypto; the quote currency's sign for forex pairs.
    var currencySymbol: String = "$"
    /// Fixed precision override (forex rates); nil picks by magnitude.
    var fractionDigits: Int? = nil

    var isPositive: Bool { change >= 0 }

    /// Amounts from this magnitude up lose their cents when compact formatting is on.
    static let compactThreshold: Double = 10_000

    /// "$762.60", "$76,509.45" (or "$76,509" when compact), "$0.0817", "€0.8710"
    func priceText(compact: Bool = false) -> String {
        currencySymbol + Self.money(price, digits: fractionDigits, compact: compact)
    }

    /// "$8.55" / "-$3.93" — positive values carry no sign, matching the reference design.
    func changeText(compact: Bool = false) -> String {
        (change < 0 ? "-" : "") + currencySymbol + Self.money(abs(change), digits: fractionDigits, compact: compact)
    }

    /// "1.13%" / "-2.60%"
    var percentText: String {
        (changePercent < 0 ? "-" : "") + abs(changePercent).formatted(Self.style(fractionDigits: 2)) + "%"
    }

    /// Two decimals for anything ≥ 1, more for small-priced coins so the change is still visible, and none
    /// for five-figure amounts when `compact` is set ("$76,520" rather than "$76,520.29").
    static func money(_ value: Double, digits: Int? = nil, compact: Bool = false) -> String {
        let magnitude = abs(value)
        if compact && magnitude >= compactThreshold {
            return value.formatted(style(fractionDigits: 0))
        }
        let digits = digits ?? (magnitude >= 1 ? 2 : magnitude >= 0.01 ? 4 : 6)
        return value.formatted(style(fractionDigits: digits))
    }

    /// US-style formatting regardless of the system locale.
    private static func style(fractionDigits: Int) -> FloatingPointFormatStyle<Double> {
        .number.precision(.fractionLength(fractionDigits)).locale(Locale(identifier: "en_US"))
    }

    // Tolerate cached quotes written before the two optional fields existed.
    private enum CodingKeys: String, CodingKey {
        case symbol, price, change, changePercent, latestTradingDay, fetchedAt, currencySymbol, fractionDigits
    }

    init(symbol: String, price: Double, change: Double, changePercent: Double, latestTradingDay: String,
         fetchedAt: Date, currencySymbol: String = "$", fractionDigits: Int? = nil) {
        self.symbol = symbol
        self.price = price
        self.change = change
        self.changePercent = changePercent
        self.latestTradingDay = latestTradingDay
        self.fetchedAt = fetchedAt
        self.currencySymbol = currencySymbol
        self.fractionDigits = fractionDigits
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try c.decode(String.self, forKey: .symbol)
        price = try c.decode(Double.self, forKey: .price)
        change = try c.decode(Double.self, forKey: .change)
        changePercent = try c.decode(Double.self, forKey: .changePercent)
        latestTradingDay = try c.decode(String.self, forKey: .latestTradingDay)
        fetchedAt = try c.decode(Date.self, forKey: .fetchedAt)
        currencySymbol = try c.decodeIfPresent(String.self, forKey: .currencySymbol) ?? "$"
        fractionDigits = try c.decodeIfPresent(Int.self, forKey: .fractionDigits)
    }
}
