import AppKit
import Combine

/// Single source of truth: watched tickers, settings, cached quotes and the refresh loop.
@MainActor
final class TickerStore: ObservableObject {
    static let defaultAlpacaKeyID = "PKQNJB2MDUDY4D5ECNB6GLBSXD"
    static let defaultAlpacaSecretKey = "EdrMQi8jLja3XXVzvAC8WPtnzeQq1nrSkg2A5JWYxb6L"
    static let defaultCoinMarketCapKey = "f39d5868aaff4f3681255a60b35f5ae7"
    static let refreshChoices = [1, 5, 15, 30, 60]
    static let quotaNote = "Alpaca 200 req/min · CoinMarketCap 10k credits/month · Frankfurter: daily ECB rates"

    @Published var tickers: [Ticker] { didSet { save() } }
    @Published var alpacaKeyID: String { didSet { save() } }
    @Published var alpacaSecretKey: String { didSet { save() } }
    @Published var coinMarketCapKey: String { didSet { save() } }
    @Published var refreshMinutes: Int { didSet { save(); restartTimer() } }
    @Published var marketHoursOnly: Bool { didSet { save() } }
    @Published var showPercentChange: Bool { didSet { save() } }
    @Published var showDollarChange: Bool { didSet { save() } }
    /// Drop the cents from five-figure amounts ("$76,520" instead of "$76,520.29").
    @Published var compactLargeAmounts: Bool { didSet { save() } }
    @Published var gainColorChoice: ColorChoice { didSet { save() } }
    @Published var lossColorChoice: ColorChoice { didSet { save() } }

    /// Keyed by `Ticker.id`.
    @Published private(set) var quotes: [String: Quote] = [:]
    @Published private(set) var errors: [String: String] = [:]
    @Published private(set) var lastUpdated: Date?
    @Published private(set) var isRefreshing = false

    let stockDirectory = SymbolDirectory()
    let coinDirectory = CoinDirectory()
    let currencyDirectory = CurrencyDirectory()

    var gainColor: NSColor { gainColorChoice.color(for: .gain) }
    var lossColor: NSColor { lossColorChoice.color(for: .loss) }

    func colorChoice(for role: ColorRole) -> ColorChoice {
        role == .gain ? gainColorChoice : lossColorChoice
    }

    func setColorChoice(_ choice: ColorChoice, for role: ColorRole) {
        if role == .gain { gainColorChoice = choice } else { lossColorChoice = choice }
    }

    private enum Keys {
        static let tickers = "tickers"
        static let legacySymbols = "symbols"
        static let alpacaKeyID = "alpacaKeyID"
        static let alpacaSecretKey = "alpacaSecretKey"
        static let coinMarketCapKey = "coinMarketCapKey"
        static let refreshMinutes = "refreshMinutes"
        static let marketHoursOnly = "marketHoursOnly"
        static let showPercentChange = "showPercentChange"
        static let showDollarChange = "showDollarChange"
        static let compactLargeAmounts = "compactLargeAmounts"
        static let gainColorChoice = "gainColorChoice"
        static let lossColorChoice = "lossColorChoice"
        static let quoteCache = "quoteCacheV2"
        static let lastUpdated = "lastUpdated"
    }

    private let defaults: UserDefaults
    private let alpaca = AlpacaClient()
    private let coinMarketCap = CoinMarketCapClient()
    private let frankfurter = FrankfurterClient()
    private var timer: Timer?
    private var inFlight = 0

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let stored = Self.decode([Ticker].self, defaults.data(forKey: Keys.tickers)) {
            tickers = stored
        } else if let legacy = defaults.stringArray(forKey: Keys.legacySymbols) {
            // Earlier builds stored plain symbols, with crypto as Alpaca "BTC/USD" pairs.
            tickers = legacy.compactMap { symbol in
                if symbol.hasSuffix("/USD") { return .unresolvedCrypto(String(symbol.dropLast(4))) }
                return symbol.contains("/") ? nil : .stock(symbol)
            }
        } else {
            tickers = [.stock("SPCX"), .stock("SPY")]
        }
        alpacaKeyID = defaults.string(forKey: Keys.alpacaKeyID) ?? Self.defaultAlpacaKeyID
        alpacaSecretKey = defaults.string(forKey: Keys.alpacaSecretKey) ?? Self.defaultAlpacaSecretKey
        coinMarketCapKey = defaults.string(forKey: Keys.coinMarketCapKey) ?? Self.defaultCoinMarketCapKey
        refreshMinutes = defaults.object(forKey: Keys.refreshMinutes) as? Int ?? 5
        marketHoursOnly = defaults.object(forKey: Keys.marketHoursOnly) as? Bool ?? true
        showPercentChange = defaults.object(forKey: Keys.showPercentChange) as? Bool ?? true
        showDollarChange = defaults.object(forKey: Keys.showDollarChange) as? Bool ?? true
        compactLargeAmounts = defaults.object(forKey: Keys.compactLargeAmounts) as? Bool ?? true
        gainColorChoice = Self.decode(ColorChoice.self, defaults.data(forKey: Keys.gainColorChoice)) ?? .system
        lossColorChoice = Self.decode(ColorChoice.self, defaults.data(forKey: Keys.lossColorChoice)) ?? .system
        lastUpdated = defaults.object(forKey: Keys.lastUpdated) as? Date
        quotes = Self.decode([String: Quote].self, defaults.data(forKey: Keys.quoteCache)) ?? [:]

        coinDirectory.onLoad = { [weak self] in self?.resolvePendingCoins() }
    }

    // MARK: Tickers

    /// Accepts "aapl", "AAPL, MSFT", "BRK.B nvda" — anything separated by commas/whitespace.
    func addStocks(_ input: String) {
        let fresh = Self.tokens(in: input)
            .map(Ticker.stock)
            .filter { !tickers.contains($0) }
        guard !fresh.isEmpty else { return }
        tickers.append(contentsOf: fresh)
        refresh(only: fresh)
    }

    func add(coin: CoinEntry) {
        let ticker = Ticker.crypto(coin)
        guard !tickers.contains(ticker) else { return }
        tickers.append(ticker)
        refresh(only: [ticker])
    }

    /// Coins typed by symbol: each resolves to the highest-ranked listing with that symbol.
    func addCrypto(_ input: String) {
        let fresh = Self.tokens(in: input)
            .map { symbol in coinDirectory.bestMatch(symbol: symbol).map(Ticker.crypto) ?? .unresolvedCrypto(symbol) }
            .filter { candidate in !tickers.contains { $0.kind == .crypto && $0.symbol == candidate.symbol } }
        guard !fresh.isEmpty else { return }
        tickers.append(contentsOf: fresh)
        refresh(only: fresh)
    }

    /// "EUR/USD", "eurusd", "GBP USD" — several pairs separated by commas.
    func addForex(_ input: String) {
        let fresh = CurrencyDirectory.parsePairs(input)
            .map { Ticker.forex($0, name: currencyDirectory.name(for: $0)) }
            .filter { !tickers.contains($0) }
        guard !fresh.isEmpty else { return }
        tickers.append(contentsOf: fresh)
        refresh(only: fresh)
    }

    func add(pair: CurrencyPair) {
        let ticker = Ticker.forex(pair, name: currencyDirectory.name(for: pair))
        guard !tickers.contains(ticker) else { return }
        tickers.append(ticker)
        refresh(only: [ticker])
    }

    func remove(_ ticker: Ticker) {
        tickers.removeAll { $0 == ticker }
        quotes[ticker.id] = nil
        errors[ticker.id] = nil
        saveCache()
    }

    /// Moves `ticker` to `index` among the *other* tickers (0 = first, `tickers.count - 1` = last).
    func move(_ ticker: Ticker, to index: Int) {
        guard let from = tickers.firstIndex(of: ticker) else { return }
        var reordered = tickers
        reordered.remove(at: from)
        let to = max(0, min(index, reordered.count))
        guard to != from else { return } // re-inserting at `from` restores the original order
        reordered.insert(ticker, at: to)
        tickers = reordered
    }

    private static func tokens(in input: String) -> [String] {
        input
            .uppercased()
            .split(whereSeparator: { $0 == "," || $0.isWhitespace })
            .map(String.init)
            .filter { !$0.isEmpty && $0.allSatisfy { $0.isLetter || $0.isNumber || ".-".contains($0) } }
    }

    /// Fills in CoinMarketCap ids for coins that were added by symbol before the map was available.
    private func resolvePendingCoins() {
        var resolved: [Ticker] = []
        tickers = tickers.map { ticker in
            guard ticker.kind == .crypto, ticker.coinID == nil,
                  let coin = coinDirectory.bestMatch(symbol: ticker.symbol) else { return ticker }
            errors[ticker.id] = nil
            let updated = ticker.resolved(with: coin)
            resolved.append(updated)
            return updated
        }
        if !resolved.isEmpty { refresh(only: resolved) }
    }

    // MARK: Refreshing

    var firstError: String? {
        tickers.lazy.compactMap { self.errors[$0.id] }.first
    }

    func startAutoRefresh() {
        restartTimer()
        stockDirectory.loadIfNeeded(keyID: alpacaKeyID, secretKey: alpacaSecretKey)
        coinDirectory.loadIfNeeded(apiKey: coinMarketCapKey)
        currencyDirectory.loadIfNeeded()
        // Serve the cache if it is younger than one refresh interval; otherwise fetch right away.
        let cacheAge = lastUpdated.map { Date().timeIntervalSince($0) } ?? .infinity
        let cacheCoversAllTickers = tickers.allSatisfy { quotes[$0.id] != nil }
        if !cacheCoversAllTickers || cacheAge > TimeInterval(refreshMinutes * 60) {
            refresh()
        }
    }

    func refresh(only subset: [Ticker]? = nil) {
        let targets = subset ?? tickers
        guard !targets.isEmpty else { return }

        inFlight += 1
        isRefreshing = true
        Task {
            let results = await fetch(targets)
            for (id, result) in results {
                switch result {
                case .success(let quote):
                    quotes[id] = quote
                    errors[id] = nil
                case .failure(let error):
                    errors[id] = error.localizedDescription
                }
            }
            lastUpdated = Date()
            inFlight -= 1
            isRefreshing = inFlight > 0
            saveCache()
        }
    }

    private func fetch(_ targets: [Ticker]) async -> [String: Result<Quote, QuoteError>] {
        let stocks = targets.filter { $0.kind == .stock }
        let coins = targets.filter { $0.kind == .crypto }
        let pairs = targets.filter { $0.kind == .forex }
        async let stockResults = fetchStocks(stocks)
        async let coinResults = fetchCoins(coins)
        async let forexResults = fetchForex(pairs)
        return await stockResults
            .merging(coinResults) { first, _ in first }
            .merging(forexResults) { first, _ in first }
    }

    private func fetchForex(_ tickers: [Ticker]) async -> [String: Result<Quote, QuoteError>] {
        guard !tickers.isEmpty else { return [:] }
        var results: [String: Result<Quote, QuoteError>] = [:]
        var valid: [Ticker] = []
        for ticker in tickers {
            if let pair = ticker.currencyPair, currencyDirectory.isKnown(pair) {
                valid.append(ticker)
            } else {
                results[ticker.id] = .failure(.unknownSymbol(ticker.symbol))
            }
        }
        guard !valid.isEmpty else { return results }
        do {
            let quotes = try await frankfurter.rates(for: valid.compactMap(\.currencyPair))
            for ticker in valid {
                results[ticker.id] = quotes[ticker.symbol].map(Result.success) ?? .failure(.unknownSymbol(ticker.symbol))
            }
        } catch {
            let quoteError = (error as? QuoteError) ?? .network(error.localizedDescription)
            valid.forEach { results[$0.id] = .failure(quoteError) }
        }
        return results
    }

    private func fetchStocks(_ stocks: [Ticker]) async -> [String: Result<Quote, QuoteError>] {
        guard !stocks.isEmpty else { return [:] }
        var results: [String: Result<Quote, QuoteError>] = [:]
        do {
            let quotes = try await alpaca.quotes(
                for: stocks.map(\.symbol),
                keyID: alpacaKeyID.trimmingCharacters(in: .whitespacesAndNewlines),
                secretKey: alpacaSecretKey.trimmingCharacters(in: .whitespacesAndNewlines)
            )
            for ticker in stocks {
                results[ticker.id] = quotes[ticker.symbol].map(Result.success) ?? .failure(.unknownSymbol(ticker.symbol))
            }
        } catch {
            let quoteError = (error as? QuoteError) ?? .network(error.localizedDescription)
            stocks.forEach { results[$0.id] = .failure(quoteError) }
        }
        return results
    }

    private func fetchCoins(_ coins: [Ticker]) async -> [String: Result<Quote, QuoteError>] {
        guard !coins.isEmpty else { return [:] }
        var results: [String: Result<Quote, QuoteError>] = [:]
        let resolved = coins.filter { $0.coinID != nil }
        // Symbols the coin map couldn't place yet (or at all).
        for ticker in coins where ticker.coinID == nil {
            results[ticker.id] = .failure(.unknownSymbol(ticker.symbol))
        }
        guard !resolved.isEmpty else { return results }
        do {
            let quotes = try await coinMarketCap.quotes(
                ids: resolved.compactMap(\.coinID),
                apiKey: coinMarketCapKey.trimmingCharacters(in: .whitespacesAndNewlines)
            )
            for ticker in resolved {
                results[ticker.id] = ticker.coinID.flatMap { quotes[$0] }.map(Result.success)
                    ?? .failure(.unknownSymbol(ticker.symbol))
            }
        } catch {
            let quoteError = (error as? QuoteError) ?? .network(error.localizedDescription)
            resolved.forEach { results[$0.id] = .failure(quoteError) }
        }
        return results
    }

    private func restartTimer() {
        timer?.invalidate()
        let interval = TimeInterval(max(1, refreshMinutes) * 60)
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.autoRefreshTick() }
        }
        timer?.tolerance = interval * 0.1
    }

    private func autoRefreshTick() {
        if marketHoursOnly && !Self.isUSMarketOpen() {
            // Crypto trades around the clock and ECB fixings land at 16:00 CET, so both keep refreshing
            // while the US stock market is closed.
            let others = tickers.filter { $0.kind != .stock }
            if !others.isEmpty { refresh(only: others) }
            return
        }
        refresh()
    }

    /// Mon–Fri 09:30–16:15 New York time (15 min grace so closing prints get picked up). Ignores holidays.
    static func isUSMarketOpen(at date: Date = Date()) -> Bool {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let parts = calendar.dateComponents([.weekday, .hour, .minute], from: date)
        guard let weekday = parts.weekday, (2...6).contains(weekday),
              let hour = parts.hour, let minute = parts.minute else { return false }
        let minutes = hour * 60 + minute
        return minutes >= 9 * 60 + 30 && minutes <= 16 * 60 + 15
    }

    // MARK: Persistence

    private func save() {
        defaults.set(try? JSONEncoder().encode(tickers), forKey: Keys.tickers)
        defaults.set(alpacaKeyID, forKey: Keys.alpacaKeyID)
        defaults.set(alpacaSecretKey, forKey: Keys.alpacaSecretKey)
        defaults.set(coinMarketCapKey, forKey: Keys.coinMarketCapKey)
        defaults.set(refreshMinutes, forKey: Keys.refreshMinutes)
        defaults.set(marketHoursOnly, forKey: Keys.marketHoursOnly)
        defaults.set(showPercentChange, forKey: Keys.showPercentChange)
        defaults.set(showDollarChange, forKey: Keys.showDollarChange)
        defaults.set(compactLargeAmounts, forKey: Keys.compactLargeAmounts)
        defaults.set(try? JSONEncoder().encode(gainColorChoice), forKey: Keys.gainColorChoice)
        defaults.set(try? JSONEncoder().encode(lossColorChoice), forKey: Keys.lossColorChoice)
    }

    private func saveCache() {
        defaults.set(try? JSONEncoder().encode(quotes), forKey: Keys.quoteCache)
        defaults.set(lastUpdated, forKey: Keys.lastUpdated)
    }

    private static func decode<T: Decodable>(_ type: T.Type, _ data: Data?) -> T? {
        data.flatMap { try? JSONDecoder().decode(type, from: $0) }
    }
}
