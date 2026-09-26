import Foundation

/// CoinMarketCap v2 latest quotes, fetched by id (up to 100 per call, one credit per call).
struct CoinMarketCapClient {
    var session: URLSession = .shared

    /// Quotes keyed by CoinMarketCap id. The 24h percentage is what CMC reports; the dollar change is
    /// backed out from it since the API doesn't return an absolute move.
    func quotes(ids: [Int], apiKey: String) async throws -> [Int: Quote] {
        guard !apiKey.isEmpty else { throw QuoteError.missingCredentials(.coinMarketCap) }
        guard !ids.isEmpty else { return [:] }

        var components = URLComponents(string: "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest")!
        components.queryItems = [
            URLQueryItem(name: "id", value: ids.map(String.init).joined(separator: ",")),
            URLQueryItem(name: "convert", value: "USD"),
        ]
        var request = URLRequest(url: components.url!)
        request.setValue(apiKey, forHTTPHeaderField: "X-CMC_PRO_API_KEY")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let status = json?["status"] as? [String: Any]
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            switch http.statusCode {
            case 401, 403: throw QuoteError.unauthorized(.coinMarketCap)
            case 429: throw QuoteError.rateLimited(.coinMarketCap)
            default:
                if let message = status?["error_message"] as? String { throw QuoteError.api(.coinMarketCap, message) }
                throw QuoteError.http(.coinMarketCap, http.statusCode)
            }
        }
        guard let coins = json?["data"] as? [String: Any] else { throw QuoteError.badResponse(.coinMarketCap) }

        var quotes: [Int: Quote] = [:]
        for (key, value) in coins {
            guard let id = Int(key), let coin = value as? [String: Any],
                  let symbol = coin["symbol"] as? String,
                  let usd = (coin["quote"] as? [String: Any])?["USD"] as? [String: Any],
                  let price = usd["price"] as? Double else { continue }
            let percent = usd["percent_change_24h"] as? Double ?? 0
            let previous = price / (1 + percent / 100)
            quotes[id] = Quote(
                symbol: symbol,
                price: price,
                change: price - previous,
                changePercent: percent,
                latestTradingDay: String((usd["last_updated"] as? String ?? "").prefix(10)),
                fetchedAt: Date()
            )
        }
        return quotes
    }
}
