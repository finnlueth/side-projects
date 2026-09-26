function optionVisableSet() {
    GetDeviceID();
    function everyTime() {
        var vendorName = document.getElementById('vendorname');
        if (vendorName != null && vendorName.innerHTML != '') {
            CallMarketIndicatorService();
            clearInterval(myInterval);
        }
    }
    var myInterval = setInterval(everyTime, 100);
}

function CallMarketIndicatorService() {
    var req = new XMLHttpRequest();
    req.responseType = 'json';
    req.open('GET', '/api/pub/owpflifo', true);
    var sorigin = document.getElementById('origin');
    var sdestination = document.getElementById('destination');
    var sflightnum = document.getElementById('flightNum');
    var origin = sorigin != null ? sorigin.innerHTML : '';
    var destination = sdestination != null ? sdestination.innerHTML : '';
    var flightnum = sflightnum != null ? sflightnum.innerHTML : '';

    req.onload = function () {
        if (this.readyState == 4 && this.status == 200) {
            try {
                var jsonResponse = JSON.stringify(req.response);
                var getmarketIndicator = JSON.parse(jsonResponse);

                if (getmarketIndicator.hasOwnProperty('GET') == true) {
                    var jsonpars = JSON.parse(getmarketIndicator.GET);
                    var marketIndicator = JSON.stringify(jsonpars.MarketIndicator);
                    var indOrigin = JSON.stringify(jsonpars.Flight.Origin);
                    var indDestination = JSON.stringify(jsonpars.Flight.Destination);
                    var indFlightNum = JSON.stringify(jsonpars.Flight.FlightNumber);
                    var internet = document.getElementById('internet-tile-div3');
                    var hamburgerinternet = document.getElementById('internet-hamburger');

                    if (marketIndicator == '"I"' &&
                        '"' + origin + '"' == indOrigin &&
                        '"' + destination + '"' == indDestination &&
                        flightnum == indFlightNum
                    ) {
                        var content = "<span id='internet-tile-span'><p class='bold' id='internet-tile-p3'>Available to buy</p><p id='internet-tile-p4'>view price details</p></span>";
                        var hameburgercontent = "<span id='internet-hamburger-tile-span'><p class='bold' id='internet-hamburger-tile-p3'>Available to buy</p><p id='internet-hamburger-tile-p4'>view price details</p></span>";
                        if (internet != null) {
                            internet.innerHTML = content;
                        }
                        if (hamburgerinternet != null) {
                            hamburgerinternet.innerHTML = hameburgercontent;
                        }
                    }
                }
            }
            catch {
                req.send(null);
            }
        }
    };
    req.send(null);
}

function GetDeviceID() {
    var req = new XMLHttpRequest();
    req.responseType = 'json';
    req.open('GET', '/portal/r/getDevice', true);
    req.onload = function () {
        if (this.readyState == 4 && this.status == 200) {
            try {
                var jsonResponse = JSON.stringify(req.response);
                var getDeviceId = JSON.parse(jsonResponse);
                var deviceid = JSON.stringify(getDeviceId.DeviceId);
                var deviceidele = document.getElementById('deviceid');
                if (deviceid != null) {
                    deviceidele.innerHTML = deviceid.substring(1).slice(0, -1);
                }
            }
            catch {
                req.send(null);
            }
        }
    };
    req.send(null);
}