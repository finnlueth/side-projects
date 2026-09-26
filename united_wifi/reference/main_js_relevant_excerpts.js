// ---- pacConfig (Panasonic) extracted from /content/home/assets/js/v1/main.js ----
var pacConfig = {
        poller: {
            delay: 10000
        },
        debugInfoEnabled: false,
        mobileAppName: 'uaportalpacstream',
        endpoints: {
            sessionApi: {
                url: '/portal/r/getAllSessionData'
            },
            signoutApi: {
                url: '/api/shim/ground/signout'
            },
            pauseApi: {
                url: '/api/shim/ground/pause'
            },
            unpauseApi: {
                url: '/api/shim/ground/unpause'
            },
            pdeAllMediaApi: {
                url: '/portal/r/getPDESupercatFlat',
                requestPageSize: 30
            },
            pdeCategoriesApi: {
                url: '/portal/r/getPDECategories'
            },
            pdeSuperCategoryMediapApi: {
                url: '/portal/r/getPDESupercatFlat',
                requestPageSize: 30
            },
            pdeSubCategoryMediapApi: {
                url: '/portal/r/getPDESubCatItems',
                requestPageSize: 30
            },
            pdeMediaItemApi: {
                url: '/portal/r/getPDEMediaItem'
            },
            registerDeviceApi: {
                url: '/portal/r/captureDevice'
            },
            flightBot: {
                portalApi: 'https://www.unitedwifi.com/api/flight/portal/v1/flifo',
                healthApi: 'https://www.unitedwifi.com/api/flight/portal/v1/health'
            },
            flightaware: {
                inflightApi: 'https://www.unitedwifi.com/flightaware/v1/'
            },
            ground: {
                urlBase: 'https://wifigroundportalqa.united.com/',
                appInfo: 'https://wifigroundportalqa.united.com/appinfo'
            }
        },
        http: {
            callback: {},
            get: {}
        },
        httpPost: {
            callback: {},
            save: {}
        },
        httpHead: {
            callback: {},
            head: {}
        },
        links: {
            switchDevice: '/portal/l/switchdevice',
            subscriberSignIn: '/portal/l/subscription',
            signIn: '/portal/l/signin',
            player: '/pde/player.html',
            purchase: '/portal/l/tiers',
            addTime: '/portal/l/internethours',
            temporarilyUnavailable: 'https://www.unitedwifi.com/content/home/unavailable.html',
            pdeDrmSupportMac: '/portal/l/supportentertainment?os=Mac&expandSysSection=true',
            pdeDrmSupportWindows: '/portal/l/supportentertainment?os=Windows&expandSysSection=true',
            pdeMobileAppDetection: '/portal/l/appdetection?',
            iOS9AppDetectionBaseUrl: 'http://www.unitedwifi.com',
            marlinPlayerPage: 'http://www.unitedwifi.com/pde/player.html',
            chaseAd: 'http://UnitedExplorerCard.com/Wi-Fi'
        },
        captcha: {
            imageSrc: '../captcha/image?captchaid=',
            audioSrc: '../captcha/audio?captchaid='
        },
        mediaItemsCarousel: {
            enableAutoScroll: true,
            autoScrollDelay: 900,
            carouselLoadRetryDelay: 10,
            carouselLoadRetryMaxAttempts: 100
        },
        vendorCode: 'PAC',
        pdeEnableNetworkCheck: false,
        internetUnavailableMsg: 'Note: connection is typically established about 10 minutes after take off.',
        undetectedBrowserMsgALL: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use the latest version of Chrome, Firefox, IE, Microsoft Edge or Safari if on a Mac. (Error 3011)',
        unsupportedBrowserOSMsgWIN: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use the latest version of Chrome, Firefox, or Microsoft Edge. (Error 3012)',
        unsupportedBrowserMsgWIN: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use the latest version of Chrome, Firefox, IE, or Microsoft Edge. (Error 3013)',
        unsupportedBrowserMsgMAC: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use the latest version of Safari, Chrome, or Firefox. (Error 3014)',
        unsupportedBrowserMsgiOS: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use Safari. (Error 3015)',
        unsupportedBrowserMsgANDROID: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use Chrome. (Error 3016)',
        unsupportedBrowserMsgiPadOS: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use Safari. (Error 3017)',
        unsupportedBrowserMsgChromeOS: 'Your browser doesn’t support personal device entertainment. ' +
            ' To watch your selection, please use Chrome. (Error 3018)',
        unsupportedOSMsgMAC: 'Your OS doesn’t support personal device entertainment. ' +
            ' To watch your selection, please make sure your device is running the latest version of macOS. (Error 3031)',
        unsupportedOSMsgWINDOWS: 'Your OS doesn’t support personal device entertainment. ' +
            ' To watch your selection, please make sure your device is running the latest version of Windows. (Error 3032)',
        unsupportedOSMsgiPAD: 'Your OS doesn’t support personal device entertainment. ' +
            ' To watch your selection, please make sure your device is running the latest version of iPad OS. (Error 3033)',
        unsupportedOSMsgChromeOS: 'Your OS doesn’t support personal device entertainment. ' +
            ' To watch your selection, please make sure your device is running the latest version of Chrome OS. (Error 3034)',
        unsupportedOSMsgiOS: 'Your OS doesn’t support personal device entertainment. ' +
            ' To watch your selection, please make sure your device is running the latest version of iOS. (Error 3035)',
        unsupportedOSMsgANDROID: 'Your OS doesn’t support personal device entertainment. ' +
            ' To watch your selection, please make sure your device is running the latest version of Android. (Error 3036)'
    };
    

// ---- MainControllerImpl.fetchGroundConnectionStatus ----
prototype.fetchGroundConnectionStatus = function () {
            var _this = this;
            var groundPortalUrl = 'https://wifigroundportal.united.com/appinfo';
            switch (this.stateDependentService.getEnvironment().toLowerCase()) {
                case 'prod':
                    groundPortalUrl = 'https://wifigroundportal.united.com/appinfo';
                    break;
                case 'stage':
                    groundPortalUrl = 'https://wifistg.united.com/appinfo';
                    break;
                case 'qa':
                    groundPortalUrl = 'https://wifigroundportalqa.united.com/appinfo';
                    break;
                case 'dev':
                    groundPortalUrl = 'https://wifidev.united.com/appinfo';
                    break;
                default:
                    groundPortalUrl = 'https://wifigroundportal.united.com/appinfo';
                    break;
            }
            fetch(groundPortalUrl, { method: 'HEAD' })
                .then(function (response) {
                if (response.ok) {
                    _this.statusService.setGroundConnectivityAvailable(true);
                    _this.statusService.setIsGlobalConnectivityAvailable();
                    _this.$scope.$apply();
                }
                else {
                    throw new Error('Not available');
                }
            })
                .catch(function (error) {
                _this.statusService.setGroundConnectivityAvailable(false);
                _this.statusService.setIsGlobalConnectivityAvailable();
                _this.$scope.$apply();
            });
        };

// ---- MainControllerImpl.fetchUiSettingsFromRedis ----
prototype.fetchUiSettingsFromRedis = function () {
            var _this = this;
            var redisUrl = 'https://www.unitedwifi.com/api/pub/uiSettings';
            fetch(redisUrl, { method: 'GET' })
                .then(function (response) {
                if (response.ok) {
                    return response.json();
                }
                else {
                    throw new Error('Not available');
                }
            })
                .then(function (data) {
                if (!data.GET) {
                    throw new Error('Missing GET property in response');
                }
                var uiSettings = JSON.parse(data.GET);
                if (!uiSettings.Settings || !uiSettings.Settings.BlockGroundAccessOnGroundHeadCallFailure || !uiSettings.Settings.BlockGroundAccessOnVendorNetworkFailure) {
                    throw new Error('Invalid UISettings JSON structure');
                }
                _this.stateDependentService.setEnvironment(uiSettings.Environment || 'prod');
                switch (_this.getVendorName().toLowerCase()) {
                    case 'thales':
                    case 'ltv':
                        _this.stateDependentService.setBlockGroundAccessOnGroundHeadCallFailure(uiSettings.Settings.BlockGroundAccessOnGroundHeadCallFailure.Thales || false);
                        _this.stateDependentService.setBlockGroundAccessOnVendorNetworkFailure(uiSettings.Settings.BlockGroundAccessOnVendorNetworkFailure.Thales || false);
                        break;
                    case 'viasat':
                    case 'via':
                        _this.stateDependentService.setBlockGroundAccessOnGroundHeadCallFailure(uiSettings.Settings.BlockGroundAccessOnGroundHeadCallFailure.Viasat || false);
                        _this.stateDependentService.setBlockGroundAccessOnVendorNetworkFailure(uiSettings.Settings.BlockGroundAccessOnVendorNetworkFailure.Viasat || false);
                        break;
                    case 'panasonic':
                    case 'pac':
                        _this.stateDependentService.setBlockGroundAccessOnGroundHeadCallFailure(uiSettings.Settings.BlockGroundAccessOnGroundHeadCallFailure.Panasonic || false);
                        _this.stateDependentService.setBlockGroundAccessOnVendorNetworkFailure(uiSettings.Settings.BlockGroundAccessOnVendorNetworkFailure.Panasonic || false);
                        break;
                    default:
                        _this.stateDependentService.setBlockGroundAccessOnGroundHeadCallFailure(false);
                        _this.stateDependentService.setBlockGroundAccessOnVendorNetworkFailure(false);
                        break;
                }
            })
                .catch(function (error) {
                _this.stateDependentService.setBlockGroundAccessOnGroundHeadCallFailure(false);
                _this.stateDependentService.setBlockGroundAccessOnVendorNetworkFailure(false);
            });
        };

// ---- StateDependentEntity.allowGroundButtonClick ----
StateDependentEntity.prototype.allowGroundButtonClick = function () {
            if (this.blockGroundAccessOnGroundHeadCallFailure && !this.statusEntityService.isGroundConnectivityAvailable()) {
                return false;
            }
            if (this.blockGroundAccessOnVendorNetworkFailure && !this.statusEntityService.isVendorConnectivityAvailable()) {
                return false;
            }
            return true;
        };