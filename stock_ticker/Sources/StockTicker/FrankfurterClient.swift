import Foundation

struct CurrencyPair: Hashable {
    let base: String
    let quote: String

    var symbol: String { "\(base)/\(quote)" }
}

/// Frankfurter (frankfurter.dev): ECB reference rates, one fixing per working day, no key needed.
/// One time-series request per base currency yields today's rate and the previous fixing for the change.
struct FrankfurterClient {
    var session: URLSession = .shared

    private static let signs: [String: String] = ["USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"]

    func rates(for pairs: [CurrencyPair]) async throws -> [String: Quote] {
        guard !pairs.isEmpty else { return [:] }
        var quotes: [String: Quote] = [:]
        for (base, group) in Dictionary(grouping: pairs, by: \.base) {
            let fetched = try await rates(base: base, quotes: group.map(\.quote))
            quotes.merge(fetched) { first, _ in first }
        }
        return quotes
    }

    private func rates(base: String, quotes: [String]) async throws -> [String: Quote] {
        // Ten calendar days comfortably covers weekends and ECB holidays, leaving ≥ 2 fixings.
        let start = Calendar(identifier: .gregorian).date(byAdding: .day, value: -10, to: Date())!
        let startText = start.formatted(.iso8601.year().month().day().dateSeparator(.dash))
        var components = URLComponents(string: "https://api.frankfurter.dev/v1/\(startText)..")!
        components.queryItems = [
            URLQueryItem(name: "base", value: base),
            URLQueryItem(name: "symbols", value: quotes.joined(separator: ",")),
        ]

        let (data, response) = try await session.data(from: components.url!)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            if http.statusCode == 404 { throw QuoteError.unknownSymbol("\(base)/\(quotes.joined(separator: ","))") }
            throw QuoteError.http(.frankfurter, http.statusCode)
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let byDate = json["rates"] as? [String: [String: Double]] else {
            throw QuoteError.badResponse(.frankfurter)
        }

        let dates = byDate.keys.sorted()
        guard let latestDate = dates.last, let latest = byDate[latestDate] else {
            throw QuoteError.badResponse(.frankfurter)
        }
        let previous = dates.dropLast().last.flatMap { byDate[$0] } ?? [:]

        var result: [String: Quote] = [:]
        for quote in quotes {
            guard let rate = latest[quote] else { continue }
            let change = previous[quote].map { rate - $0 } ?? 0
            let percent = previous[quote].map { change / $0 * 100 } ?? 0
            result["\(base)/\(quote)"] = Quote(
                symbol: "\(base)/\(quote)",
                price: rate,
                change: change,
                changePercent: percent,
                latestTradingDay: latestDate,
                fetchedAt: Date(),
                currencySymbol: Self.signs[quote] ?? "\(quote) ",
                fractionDigits: rate >= 20 ? 2 : 4
            )
        }
        return result
    }
}
