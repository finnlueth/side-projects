#!/usr/bin/env bash
# united_wifi_diag.sh — trace the United Wi-Fi captive portal and get to the
# ground portal's product page directly (where the T-Mobile option lives).
#
# Works offline-ish: only talks to the on-board portal (www.unitedwifi.com) and
# United's ground portal (wifigroundportal.united.com), which the walled garden
# allows before you have paid. Needs only curl + python3 (both ship with macOS).
#
# Usage:
#   ./united_wifi_diag.sh            # full run
#   ./united_wifi_diag.sh --no-post  # on-plane diagnostics only, skip ground POST
#   ./united_wifi_diag.sh --probe    # also probe a few guessed ground paths
#
# Output goes to ./runs/<timestamp>/ — read summary.txt first, then open
# open_ground_portal.html in your browser to land on the real products page.

set -u
cd "$(dirname "$0")"

PLANE="https://www.unitedwifi.com"
UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'
DO_POST=1; DO_PROBE=0
for a in "$@"; do
  case "$a" in
    --no-post) DO_POST=0 ;;
    --probe)   DO_PROBE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  esac
done

RUN="runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
SUM="$RUN/summary.txt"
JAR="$RUN/cookies.txt"

log() { printf '%s\n' "$*" | tee -a "$SUM"; }
hr()  { log ""; log "==================== $* ===================="; }

# curl wrapper: -k because the captive portal may MITM/redirect TLS before login
CURL() { curl -sS -k -A "$UA" -c "$JAR" -b "$JAR" --max-time 25 "$@"; }

# ---------------------------------------------------------------- 1. session
hr "1. On-plane session state  (GET $PLANE/portal/r/getAllSessionData)"
CURL -o "$RUN/session.json" -w 'http=%{http_code}\n' "$PLANE/portal/r/getAllSessionData" | tee -a "$SUM"
python3 - "$RUN/session.json" <<'EOF' | tee -a "$SUM"
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("  !! could not parse session.json:",e); sys.exit()
i=d.get("internet",{}); f=d.get("flifo",{}); p=d.get("metaData",{}).get("properties",{})
print(f"  vendor            : {d.get('VendorName')} ({d.get('VendorCode')})   portal v{d.get('version')}  deviceCAS={d.get('deviceCAS')}")
print(f"  flight            : UA{f.get('flightNumber')}  {f.get('originAirportCode')} -> {f.get('destinationAirportCode')}   tail {f.get('tailNumber')}  nose {f.get('noseNumber')}  {f.get('aircraftModel')}")
print(f"  IsLoggedIn        : {d.get('IsLoggedIn')}")
print(f"  internet          : accessType={i.get('accessType')!r} tier={i.get('tier')!r} desc={i.get('tierDescription')!r} order={i.get('orderState')} fulfil={i.get('fulfilmentState')} timeRemaining={i.get('timeRemaining')}")
print(f"  offers internet   : {d.get('flightOffersInternetService')}   connection available: {d.get('internetConnectionIsAvailable')}   subscription avail: {d.get('isSubscriptionAvailable')}")
print(f"  device            : MAC={p.get('Mac')}  ip={p.get('IpAddress')}  framedIp={p.get('FramedIp')}  MileagePlus={p.get('MileagePlusId')}")
print(f"  metaData.tags     : {d.get('metaData',{}).get('tags')}")
print(f"  coverage msg      : {d.get('CoverageMessage')!r}  showCoverageArea={d.get('ShowCoverageArea')}")
EOF

# ---------------------------------------------------------------- 2. flifo / market
hr "2. Flight + market indicator  (GET $PLANE/api/pub/owpflifo)"
CURL -o "$RUN/owpflifo.json" -w 'http=%{http_code}\n' "$PLANE/api/pub/owpflifo" | tee -a "$SUM"
python3 - "$RUN/owpflifo.json" <<'EOF' | tee -a "$SUM"
import json,sys
try:
    d=json.load(open(sys.argv[1])); d=json.loads(d["GET"]) if "GET" in d else d
    fl=d.get("Flight",{})
    print(f"  MarketIndicator   : {d.get('MarketIndicator')!r}   (D = domestic, I = international)")
    print(f"  Flight            : {fl.get('CarrierCode')}{fl.get('FlightNumber')} {fl.get('Origin')}->{fl.get('Destination')}  tail {fl.get('Tail')}  IntlInd={fl.get('IntlInd')} LongHaul={fl.get('LongHaulInd')}  state={fl.get('FlightState')}")
except Exception as e: print("  !! could not parse owpflifo:",e)
EOF

# ---------------------------------------------------------------- 3. ui settings
hr "3. Portal feature flags  (GET $PLANE/api/pub/uiSettings)"
CURL -o "$RUN/uiSettings.json" -w 'http=%{http_code}\n' "$PLANE/api/pub/uiSettings" | tee -a "$SUM"
python3 - "$RUN/uiSettings.json" <<'EOF' | tee -a "$SUM"
import json,sys
try:
    d=json.load(open(sys.argv[1])); d=json.loads(d["GET"]) if "GET" in d else d
    print(json.dumps(d.get("Settings",d),indent=4)); print("  Environment:",d.get("Environment"))
    print("  -> if BlockGroundAccessOnGroundHeadCallFailure.<vendor> is true, ALL ground links (Wi-Fi tile,")
    print("     subscriber sign-in, day pass, switch device) are disabled when step 4 fails in the browser.")
except Exception as e: print("  !! could not parse uiSettings:",e)
EOF

# ---------------------------------------------------------------- 4. ground gate
hr "4. Ground-portal reachability gate  (HEAD https://wifigroundportal.united.com/appinfo)"
log "  (this is the exact check the home page runs; a non-2xx here disables every ground link)"
CURL -I -o "$RUN/appinfo_head.txt" -w 'http=%{http_code} time=%{time_total}s\n' "https://wifigroundportal.united.com/appinfo" 2>&1 | tee -a "$SUM"
grep -i -E '^(HTTP|Access-Control|Server|Location)' "$RUN/appinfo_head.txt" 2>/dev/null | sed 's/^/  /' | tee -a "$SUM"

# ---------------------------------------------------------------- 5. gotoground form
hr "5. Hand-off form  (GET $PLANE/portal/l/tiers -> gotoground?path=/login/products)"
CURL -L -o "$RUN/gotoground.html" -w 'final=%{url_effective} http=%{http_code}\n' "$PLANE/portal/l/tiers" | tee -a "$SUM"
python3 - "$RUN/gotoground.html" "$RUN/form.json" <<'EOF' | tee -a "$SUM"
import re,json,sys,uuid
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
m=re.search(r"action='([^']+)'",s)
if not m: print("  !! no hand-off form found (are you connected to the United_Wi-Fi SSID?)"); sys.exit()
fields=dict(re.findall(r"name='([^']+)' value='([^']*)'",s))
cas=re.search(r"window\.DeviceCAS\s*=\s*'([^']*)'",s); ver=re.search(r"window\.Version\s*=\s*'([^']*)'",s)
fields["DeviceCAS"]=cas.group(1) if cas else ""; fields["Version"]=ver.group(1) if ver else ""
fields["correlationId"]=str(uuid.uuid4())
json.dump({"action":m.group(1),"fields":fields},open(sys.argv[2],"w"),indent=2)
print("  action:",m.group(1))
for k,v in fields.items(): print(f"  {k:14s}= {v}")
EOF

# ---------------------------------------------------------------- 6. replay POST
if [ "$DO_POST" = 1 ] && [ -f "$RUN/form.json" ]; then
  hr "6. Replay the hand-off POST -> ground products page"
  # write the multipart fields to a curl config file (-K) to avoid shell quoting issues
  ACTION=$(python3 - "$RUN/form.json" "$RUN/curl_form.cfg" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
with open(sys.argv[2],"w") as cfg:
    for k,v in d["fields"].items():
        cfg.write('form = "%s=%s"\n' % (k, v.replace('\\','\\\\').replace('"','\\"')))
print(d["action"])
EOF
)
  CURL -K "$RUN/curl_form.cfg" -L -D "$RUN/ground_headers.txt" -o "$RUN/ground_products.html" \
       -w 'final=%{url_effective} http=%{http_code} size=%{size_download}\n' "$ACTION" | tee -a "$SUM"
  python3 - "$RUN/ground_products.html" <<'EOF' | tee -a "$SUM"
import re,html,sys
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
t=re.sub(r'<(script|style).*?</\1>','',s,flags=re.S); t=html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',t)))
print("  title      :", (re.search(r'<title>(.*?)</title>',s,re.S|re.I) or [None,'?'])[1].strip())
print("  T-Mobile   :", len(re.findall(r'\bt[\s\-_]?mobile\b',s,re.I)), "mention(s)   (0 = ground portal offered no T-Mobile option for this flight/device)")
for kw in ['mobile','carrier','partner','free','complimentary','starlink','mileageplus','sign in','verify','phone']:
    n=len(re.findall(kw,t,re.I))
    if n: print(f"  '{kw}': {n}")
print("  forms      :"); [print("    ",a) for a in dict.fromkeys(re.findall(r'<form[^>]*action=[\"\x27]([^\"\x27]+)',s,re.I))]
print("  links      :"); [print("    ",a) for a in list(dict.fromkeys(re.findall(r'href=[\"\x27]((?:https?:)?/[^\"\x27#]+)',s,re.I)))[:40]]
print("  buttons    :"); [print("    ",re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>','',b))).strip()[:80]) for b in re.findall(r'<button[^>]*>(.*?)</button>',s,re.S|re.I)[:30]]
print("  submit vals:"); [print("    ",v) for v in re.findall(r'<input[^>]+type=[\"\x27]submit[\"\x27][^>]*value=[\"\x27]([^\"\x27]+)',s,re.I)]
print("  --- visible text (first 1500 chars) ---"); print("  "+t[:1500])
EOF
fi

# ---------------------------------------------------------------- 6b. T-Mobile route
if [ "$DO_POST" = 1 ] && [ -f "$RUN/curl_form.cfg" ]; then
  hr "6b. T-Mobile eligibility  (POST hand-off form -> /login/tmobile)"
  log "  (confirmed 2026-09-21: this route exists; ineligible flights 302 -> /TMobile/v2/Unavailable)"
  BASE=${ACTION%/login*}
  CURL -K "$RUN/curl_form.cfg" -L -D "$RUN/tmobile_headers.txt" -o "$RUN/tmobile.html" \
       -w 'final=%{url_effective} http=%{http_code} size=%{size_download}\n' "$BASE/login/tmobile" | tee -a "$SUM"
  grep -i -E '^Location' "$RUN/tmobile_headers.txt" | sed 's/^/  /' | tee -a "$SUM"
  python3 - "$RUN/tmobile.html" <<'EOF' | tee -a "$SUM"
import re,html,sys
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
t=re.sub(r'<(script|style).*?</\1>','',s,flags=re.S); t=html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',t)))
i=t.find('Feedback',t.find('Feedback')+1)   # skip the two nav menus
print("  forms      :", re.findall(r'<form[^>]*action=[\"\x27]([^\"\x27]+)',s,re.I))
print("  inputs     :", [(m[0],m[1]) for m in re.findall(r'<input[^>]+name=[\"\x27]([^\"\x27]+)[\"\x27][^>]*type=[\"\x27]([^\"\x27]+)',s,re.I)][:15])
print("  --- visible text ---"); print("  "+(t[i+8:i+1200] if i>0 else t[:1200]))
if 'not available on this flight' in t: print("\n  ==> VERDICT: ground portal says T-Mobile Wi-Fi is NOT offered on this flight (server-side eligibility).")
EOF
fi

# ---------------------------------------------------------------- 7. browser replay page
if [ -f "$RUN/form.json" ]; then
  hr "7. Browser replay page"
  python3 - "$RUN/form.json" "$RUN/open_ground_portal.html" <<'EOF' | tee -a "$SUM"
import json,sys,html
d=json.load(open(sys.argv[1])); a=d["action"]; f=d["fields"]
def form(action,label,auto=False):
    inp="".join(f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">' for k,v in f.items())
    return f'<form method="post" enctype="multipart/form-data" action="{html.escape(action)}"{" id=auto" if auto else ""}>{inp}<button>{html.escape(label)}</button></form>'
base=a.rsplit("/login",1)[0]
page=f"""<!doctype html><meta charset=utf-8><title>United ground portal — direct</title>
<body style="font:16px system-ui;max-width:640px;margin:40px auto;line-height:1.5">
<h2>United Wi-Fi ground portal — direct hand-off</h2>
<p>These forms replay exactly what <code>/portal/l/gotoground</code> does, bypassing the on-board home page JS.</p>
{form(a,"Products / Wi-Fi plans  (" + a + ")",True)}
{form(base+"/login/tmobile","T-Mobile free Wi-Fi  (/login/tmobile)")}
{form(base+"/login","Sign in  (/login)")}
{form(base+"/login/subscription","Subscriber sign-in  (/login/subscription)")}
{form(base+"/login/freeproduct","Free messaging  (/login/freeproduct)")}
{form(base+"/login/switchdevice","Switch device  (/login/switchdevice)")}
<p style="color:#666">Fields: {html.escape(", ".join(f"{k}={v}" for k,v in f.items()))}</p>
<script>if(location.hash==='#auto'){{document.getElementById('auto').submit()}}</script>
</body>"""
open(sys.argv[2],"w").write(page)
print("  wrote", sys.argv[2])
print("  -> open it in your browser and click 'Products'.  Append #auto to the file URL to auto-submit.")
EOF
fi

# ---------------------------------------------------------------- 8. optional probes
if [ "$DO_PROBE" = 1 ] && [ -f "$RUN/form.json" ]; then
  hr "8. Probe guessed ground-portal paths (these are GUESSES, 404s are expected)"
  BASE=$(python3 -c "import json;print(json.load(open('$RUN/form.json'))['action'].rsplit('/login',1)[0])")
  for p in /appinfo /login /login/products /login/tmobile /login/t-mobile /login/partner /login/partners /login/carrier /login/sponsored /login/freewifi /login/complimentary /login/mileageplus /api/products /api/partners /health; do
    printf '  %-28s ' "$p" | tee -a "$SUM"
    CURL -o /dev/null -w '%{http_code} %{redirect_url}\n' "$BASE$p" | tee -a "$SUM"
  done
fi

hr "done"
log "Artifacts in $RUN/  — read summary.txt, open open_ground_portal.html"
