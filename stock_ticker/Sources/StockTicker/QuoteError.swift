import Foundation

enum QuoteProvider {
    case alpaca, coinMarketCap, frankfurter

    var name: String {
        switch self {
        case .alpaca: return "Alpaca"
        case .coinMarketCap: return "CoinMarketCap"
        case .frankfurter: return "Frankfurter"
        }
    }
}

enum QuoteError: LocalizedError {
    case missingCredentials(QuoteProvider)
    case unauthorized(QuoteProvider)
    case rateLimited(QuoteProvider)
    case unknownSymbol(String)
    case badResponse(QuoteProvider)
    case http(QuoteProvider, Int)
    case api(QuoteProvider, String)
    case network(String)

    var errorDescription: String? {
        switch self {
        case .missingCredentials(let p): return "No \(p.name) credentials set."
        case .unauthorized(let p): return "\(p.name) rejected the credentials."
        case .rateLimited(.alpaca): return "Alpaca rate limit reached (200 requests/min)."
        case .rateLimited(.coinMarketCap): return "CoinMarketCap rate limit or monthly credits reached."
        case .rateLimited(let p): return "\(p.name) rate limit reached."
        case .unknownSymbol(let symbol): return "No quote found for \(symbol)."
        case .badResponse(let p): return "Unexpected response from \(p.name)."
        case .http(let p, let code): return "\(p.name) returned HTTP \(code)."
        case .api(let p, let message): return "\(p.name): \(message)"
        case .network(let message): return message
        }
    }
}
