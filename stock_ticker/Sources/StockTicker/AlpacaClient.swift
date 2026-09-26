import Foundation

/// Alpaca Market Data v2 stock snapshots: one request covers every symbol.
/// Price is the latest trade (falls back to the daily close); change is measured against the previous
/// daily close, which is how Alpaca's own UI reports it.
struct AlpacaClient {
    var session: URLSession = .shared

    func quotes(for symbols: [String], keyID: String, secretKey: String) async throws -> [String: Quote] {
        guard !keyID.isEmpty, !secretKey.isEmpty else { throw QuoteError.missingCredentials(.alpaca) }
        guard !symbols.isEmpty else { return [:] }

        var components = URLComponents(string: "https://data.alpaca.markets/v2/stocks/snapshots")!
        components.queryItems = [URLQueryItem(name: "symbols", value: symbols.joined(separator: ","))]
        var request = URLRequest(url: components.url!)
        request.setValue(keyID, forHTTPHeaderField: "APCA-API-KEY-ID")
        request.setValue(secretKey, forHTTPHeaderField: "APCA-API-SECRET-KEY")

        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse {
            switch http.statusCode {
            case 200..<300: break
            case 401, 403: throw QuoteError.unauthorized(.alpaca)
            case 429: throw QuoteError.rateLimited(.alpaca)
            default: throw QuoteError.http(.alpaca, http.statusCode)
            }
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw QuoteError.badResponse(.alpaca)
        }

        var quotes: [String: Quote] = [:]
        for (symbol, value) in json {
            // Unknown symbols are simply absent from the response; the caller reports those.
            guard let snapshot = value as? [String: Any] else { continue }
            let latestTrade = snapshot["latestTrade"] as? [String: Any]
            let dailyBar = snapshot["dailyBar"] as? [String: Any]
            let previousBar = snapshot["prevDailyBar"] as? [String: Any]
            guard
                let price = (latestTrade?["p"] ?? dailyBar?["c"]) as? Double,
                let previousClose = previousBar?["c"] as? Double, previousClose > 0
            else { continue }

            let change = price - previousClose
            quotes[symbol] = Quote(
                symbol: symbol,
                price: price,
                change: change,
                changePercent: change / previousClose * 100,
                latestTradingDay: String((dailyBar?["t"] as? String ?? "").prefix(10)),
                fetchedAt: Date()
            )
        }
        return quotes
    }
}
