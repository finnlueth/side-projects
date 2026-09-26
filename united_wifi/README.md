# United Wi-Fi portal — T-Mobile free Wi-Fi investigation

Investigated 2026-09-21 on UA194 SFO→MUC (Boeing 777-200ER, tail N785UA,
Panasonic connectivity, portal v3.7.0). Goal: find out why the T-Mobile
"free Wi-Fi" option didn't show up in the captive portal UI, and find a
repeatable way to get to it directly.

**Short version — confirmed:** the T-Mobile option is *not* part of the
on-board portal at all. `www.unitedwifi.com` is a static shell running on the
aircraft; every purchase/login decision is made by United's **ground portal**
at `wifigroundportal.united.com`, which the on-board page hands you off to
with a hidden auto-submitting POST form. The ground portal has a dedicated
T-Mobile flow at **`POST /login/tmobile` → `/TMobile/v2/…`**, and on this
flight it answered:

> *"T-Mobile inflight connectivity is not available on this flight. You can
> still activate free messaging or purchase a Wi-Fi session."*
> (`302 → /TMobile/v2/Unavailable`, sample in `reference/`)

So nothing was "broken" client-side: the button is simply not rendered when
the ground portal's server-side eligibility check fails for the flight.
`united_wifi_diag.sh` reproduces the whole chain and prints this verdict.

---

## How the portal works

```
 ┌──────────────── aircraft (walled garden) ────────────────┐   ┌───── internet ─────┐
 │  www.unitedwifi.com  (Panasonic "PAC Web Server")        │   │ wifigroundportal.  │
 │                                                          │   │   united.com       │
 │  /content/home/index.html   static AngularJS 1.4 shell   │   │   (Kestrel/.NET)   │
 │     ├─ GET /portal/r/getAllSessionData   session/flight  │   │                    │
 │     ├─ GET /portal/r/getDevice           your MAC/IP     │   │  /appinfo          │
 │     ├─ GET /api/pub/owpflifo             MarketIndicator │   │  /login            │
 │     ├─ GET /api/pub/uiSettings           feature flags   │   │  /login/products   │◄─ plans + (T-Mobile?)
 │     └─ HEAD https://wifigroundportal.united.com/appinfo ─┼───┼─►gate check        │
 │                                                          │   │  /login/subscription│
 │  /portal/l/tiers ─302─► /portal/l/gotoground?path=…      │   │  /login/freeproduct │
 │       renders a hidden <form method=post> and submits ───┼───┼─►/login/switchdevice│
 └──────────────────────────────────────────────────────────┘   └────────────────────┘
```

### The hand-off form (`/portal/l/gotoground?path=/login/products`)

```html
<form method="post" enctype="multipart/form-data"
      action="https://wifigroundportal.united.com/login/products">
  tailNumber   = N785UA
  vendor       = Panasonic
  deviceId     = <your MAC>        <!-- your Wi-Fi MAC -->
  FlightNumber = 194
  Origin       = SFO
  Destination  = MUC
  DeviceCAS    = 7                        <!-- filled by JS from window.DeviceCAS -->
  Version      = 3.7.0                    <!-- filled by JS from window.Version -->
  correlationId= <random uuid v4>         <!-- filled by JS -->
</form>
```

The device is identified by **MAC address**, not cookies — `curl` from the same
laptop sees the same logged-in session the browser does. Sample saved in
`reference/gotoground_page_sample.html`.

### On-board link → ground path map

| On-board link            | Redirects to                                  | Ground portal action                                   |
|--------------------------|-----------------------------------------------|--------------------------------------------------------|
| Wi-Fi tile / Add time    | `/portal/l/tiers`, `/portal/l/internethours`  | `POST https://wifigroundportal.united.com/login/products` |
| Sign in                  | `/portal/l/signin`                            | `POST …/login`                                          |
| Subscriber / Day Pass    | `/portal/l/subscription`                      | `POST …/login/subscription`                             |
| Messaging (free)         | `gotoground?path=/login/freeproduct`          | `POST …/login/freeproduct`                              |
| Switch device            | `/portal/l/switchdevice`                      | `POST …/login/switchdevice`                             |
| Sign out / pause         | `/api/shim/ground/signout`, `…/pause`, `…/unpause` | (on-board shim)                                    |
| **T-Mobile free Wi-Fi**  | *(no on-board link exists in portal v3.7.0)*  | `POST …/login/tmobile` → `/TMobile/v2/…` (server-rendered) |

Ground portal routes observed (2026-09-21): `/appinfo` 200, `/health` 200,
`/login` (302 home w/o form), `/login/products` → `/Products`,
`/login/tmobile` → `/TMobile/v2/Unavailable`. Every `/TMobile/*` path
(`v2/Index`, `v2/Login`, `v2/Verify`, …) also 302s to `Unavailable` when the
flight is ineligible, so the check is session-wide, not per-page. The ground
`/js/main.js` bundle contains no T-Mobile logic — the flow is server-side
(.NET/Kestrel), so the eligibility criteria can't be read from the client.

### The gate that hides/disables links (`main.js`)

On page load the home page does:

1. `GET /api/pub/uiSettings` → on this flight:
   ```json
   "BlockGroundAccessOnGroundHeadCallFailure": {"Thales": true, "Panasonic": true, "Viasat": true},
   "BlockGroundAccessOnVendorNetworkFailure":  {"Thales": false, "Panasonic": false, "Viasat": false}
   ```
2. `fetch('https://wifigroundportal.united.com/appinfo', {method:'HEAD'})` from the **browser**.
3. If that HEAD fails and the flag for your vendor is `true` (it is, for Panasonic),
   `allowGroundButtonClick()` returns `false` → every ground link gets
   `ng-click="$event.preventDefault()"` + `aria-disabled`, and the *Switch device*
   link (`ng-if="main.isGlobalConnectivityAvailable()"`) is removed from the DOM
   entirely. Tiles still *look* clickable but do nothing.

Excerpts in `reference/main_js_relevant_excerpts.js`.

Things that make the browser HEAD fail while `curl` succeeds: ad/tracker
blockers or privacy extensions blocking `united.com`, a VPN/DNS-over-HTTPS
client, Private Relay, or the ground link simply being down for a few minutes
after takeoff. On this flight the HEAD returned `200 OK` (Kestrel), so the gate
was open.

## Why the T-Mobile button was missing (confirmed 2026-09-21)

There is **zero** T-Mobile logic on the aircraft — searched `index.html`,
`partners.html`, `main.js` (534 KB), `wifipartner.js`, `roadblock.js`,
`amanager.js`. The on-board portal (v3.7.0) doesn't even have a link to
`/login/tmobile`; when the button exists it is rendered by the ground portal's
`/Products` page. On UA194 the ground portal:

- rendered `/Products` with only *"Upgrade your plan — Wi-Fi $16.99 or 1700
  miles for full flight"* (you already held a plan, so only the upgrade
  product was offered), no T-Mobile option;
- answered `POST /login/tmobile` (with the complete hand-off form) with
  `302 → /TMobile/v2/Unavailable`: *"T-Mobile inflight connectivity is not
  available on this flight."*

The decision is server-side and we can't read the rule, but the inputs it has
are exactly the posted fields plus the device session. Most likely drivers,
by how the message is phrased ("on this flight"):

1. **Route / market.** `Origin=SFO Destination=MUC`, `MarketIndicator: "I"`
   (international), `IntlInd: true`. Compare step 6b on a domestic flight.
2. **Aircraft / connectivity vendor.** `vendor=Panasonic`, `tailNumber=N785UA`
   (777-200ER). The perk may be limited to certain systems (e.g. Starlink or
   Viasat aircraft). Compare step 6b on a different fleet type.
3. **Device already has an active product.** Less likely given the wording,
   but on the next flight run the script *before* buying anything so the
   products page shows the un-purchased state.

Separately, the **on-board gate** (section above) is the failure mode where
tiles look clickable but do nothing / "Switch device" vanishes. It was open on
this flight (`HEAD /appinfo → 200`); if it isn't, `open_ground_portal.html`
bypasses it entirely.

## Repeat on the next flight

Requirements: macOS `curl` + `python3` (Xcode CLT). No network besides the
United SSID.

```sh
cd united_wifi
./united_wifi_diag.sh              # full trace incl. ground POST
./united_wifi_diag.sh --no-post    # on-plane only (what was verified 2026-09-21)
./united_wifi_diag.sh --probe      # + guessed ground paths (/login/tmobile etc.)
```

Everything lands in `runs/<timestamp>/`:

| File | What |
|------|------|
| `summary.txt` | Human-readable trace of all steps |
| `session.json`, `owpflifo.json`, `uiSettings.json` | Raw on-board API responses |
| `appinfo_head.txt` | Ground gate HEAD response |
| `gotoground.html`, `form.json` | The hand-off form and parsed fields |
| `ground_products.html`, `ground_headers.txt` | The real products page from the ground portal |
| `tmobile.html`, `tmobile_headers.txt` | Result of `POST /login/tmobile` — the T-Mobile eligibility verdict |
| `open_ground_portal.html` | **Open in a browser** — buttons that replay the POST to `/login/products`, `/login/tmobile`, `/login`, `/login/subscription`, `/login/freeproduct`, `/login/switchdevice`. Add `#auto` to the URL to auto-submit products. |

`runs/` is git-ignored: it contains your MAC address and MileagePlus number.

### Manual one-liner (if the script is unavailable)

```sh
# 1. read the hand-off form the plane wants you to submit
curl -sk https://www.unitedwifi.com/portal/l/tiers -L | grep -E "action=|name="
# 2. replay it
curl -sk -L -F tailNumber=N785UA -F vendor=Panasonic -F 'deviceId=<your MAC>' \
  -F FlightNumber=194 -F Origin=SFO -F Destination=MUC -F DeviceCAS=7 -F Version=3.7.0 \
  -F correlationId=$(uuidgen | tr A-Z a-z) \
  https://wifigroundportal.united.com/login/products -o products.html
grep -i -c 't-\?mobile' products.html
# 3. ask the T-Mobile route directly (same fields) — follow the redirect and read the message
curl -sk -L -F tailNumber=N785UA -F vendor=Panasonic -F 'deviceId=<your MAC>' \
  -F FlightNumber=194 -F Origin=SFO -F Destination=MUC -F DeviceCAS=7 -F Version=3.7.0 \
  -F correlationId=$(uuidgen | tr A-Z a-z) \
  https://wifigroundportal.united.com/login/tmobile -o tmobile.html -w '%{url_effective}\n'
```

Get your MAC and the current DeviceCAS/Version from
`curl -sk https://www.unitedwifi.com/portal/r/getAllSessionData`.

## Files

- `united_wifi_diag.sh` — the repeatable trace.
- `reference/` — snapshots from 2026-09-21: hand-off page, session JSON,
  uiSettings JSON, `env.js`, `wifipartner.js`, relevant `main.js` excerpts
  (ground gate, uiSettings parsing, Panasonic endpoint config), and the
  `/TMobile/v2/Unavailable` page as served for UA194.
