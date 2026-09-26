# SocialFocus Custom

A personal fork of [SocialFocus: Hide Distractions](https://socialfocus.app) v7.2
(Firefox build, extracted from the installed `.xpi`), tweaked for how I actually use it.

## What changed

### Daily limit only applies to feed pages

The countdown timer, the time counting and the "Time is up" block now only apply to a
site's **primary / feed pages** — e.g. the YouTube home page, the X/LinkedIn home feed,
Reddit listings. Watching a specific video, reading a post or a profile is never counted
or blocked.

- Which URLs count as a feed page is defined per site in `feedPaths` in
  `global/storageConstants.js` (regexes on `location.pathname`; an empty list means the
  whole site counts — only Gmail uses that).
- Sites are SPAs, so `website/scripts/Specials/dailyLimit/blockingSiteByTimer.js` watches
  the URL and re-evaluates on every navigation: leaving the feed stops the counter and
  hides the on-site countdown; coming back resumes it.
- When time is up, the feed page is covered by an opaque overlay instead of wiping the
  page, so navigating away (back button, search, a direct link) brings the site straight
  back.
- The daily counter now also resets correctly at midnight in a tab that stays open.

### Free time (new setting)

More menu → **Free Time**. During the daily window (default **7:00 PM – 11:00 PM**,
enabled by default) every feature is paused: distractions are shown again, daily limits
do not count down and nothing is blocked. The stored settings are untouched, so
everything comes back when the window ends. The popup shows a banner while free time is
active. Windows may wrap past midnight (e.g. 22:00 – 02:00).

Implementation: `website/scripts/FreeTime/freeTime.js` forces the
`socialFocus_global_enable` attribute (which all site CSS is gated on) to `false` during
the window and lets the daily-limit logic re-evaluate when the state flips.

### Removed: account, sign in, PRO, cloud sync

All account / PRO / verification screens, the login-state checks and the cloud syncing
code (`serverMethods.js`, `syncMethods.js`) are gone. The extension makes no network
requests anymore.

### Small fixes picked up along the way

- The keyboard shortcut toggle called a popup-only function from a content script and
  could never have worked; it now toggles the stored state.
- Changing the daily limit from the popup no longer throws in the content script.
- Blocking-page text colour was an invalid CSS value (`dark`).

## Carrying settings over

`migration/seed.json` holds settings copied out of the original install's `storage.local`
(read from the Zen profile's IndexedDB with `moz-idb-edit`). `extension/scripts/background.js`
writes them into storage once per `seedId` when the extension is installed or updated.
Delete the file (or bump `seedId` with new contents) once it has done its job.
`build.sh` also leaves `dist/settings-import-string.txt`, the same data in the format
the *Import / Export* screen accepts, as a manual fallback.

## Install

The fork has its own add-on id (`socialfocus-custom@lueth.net`) and name, so it can be
installed next to the original. Settings are carried over by the seed described above.

- Temporary (until browser restart): `about:debugging#/runtime/this-firefox` →
  *Load Temporary Add-on…* → pick `manifest.json`.
- Permanent: build with `./build.sh` and either sign the `.xpi` with
  `web-ext sign` (AMO self-distribution) or set `xpinstall.signatures.required`
  to `false` in `about:config` if the browser allows it.
