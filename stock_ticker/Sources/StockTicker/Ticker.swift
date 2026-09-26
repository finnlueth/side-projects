import Foundation

/// One entry in the bar: a stock/ETF quoted by Alpaca, a coin quoted by CoinMarketCap, or a currency
/// pair quoted by Frankfurter (ECB reference rates).
struct Ticker: Codable, Hashable, Identifiable {
    enum Kind: String, Codable, CaseIterable {
        case stock, crypto, forex

        var title: String {
            switch self {
            case .stock: return "Stocks"
            case .crypto: return "Crypto"
            case .forex: return "Forex"
            }
        }
    }

    let kind: Kind
    let symbol: String
    /// CoinMarketCap id. Symbols aren't unique there ("BTC" matches four listings); the id is.
    let coinID: Int?
    let name: String?

    /// Stable identity used for quotes, errors and reordering.
    var id: String {
        switch kind {
        case .stock: return "stock:\(symbol)"
        case .crypto: return "crypto:" + (coinID.map(String.init) ?? symbol)
        case .forex: return "forex:\(symbol)"
        }
    }

    /// "EUR/USD" → (EUR, USD) for forex tickers.
    var currencyPair: CurrencyPair? {
        guard kind == .forex else { return nil }
        let parts = symbol.split(separator: "/")
        guard parts.count == 2 else { return nil }
        return CurrencyPair(base: String(parts[0]), quote: String(parts[1]))
    }

    static func stock(_ symbol: String) -> Ticker {
        Ticker(kind: .stock, symbol: symbol, coinID: nil, name: nil)
    }

    static func crypto(_ coin: CoinEntry) -> Ticker {
        Ticker(kind: .crypto, symbol: coin.symbol, coinID: coin.id, name: coin.name)
    }

    /// A coin typed by symbol before the directory could resolve it; the store fills in the id later.
    static func unresolvedCrypto(_ symbol: String) -> Ticker {
        Ticker(kind: .crypto, symbol: symbol, coinID: nil, name: nil)
    }

    func resolved(with coin: CoinEntry) -> Ticker { .crypto(coin) }

    static func forex(_ pair: CurrencyPair, name: String?) -> Ticker {
        Ticker(kind: .forex, symbol: pair.symbol, coinID: nil, name: name)
    }
}
