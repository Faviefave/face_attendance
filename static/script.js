
var socket


if (cv.getBuildInformation)
      {
          console.log(cv.getBuildInformation());
          console.log("Loaded CV")
          onloadCallback();
      }
      else
      {
          // WASM
          cv['onRuntimeInitialized']=()=>{
              //console.log(cv.getBuildInformation());
              onloadCallback();
          }
      }


  function onloadCallback(){
    // var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);
        socket = io();
        socket.on('connect', function () {
            console.log("Connected...!", socket.connected)
        });

        var canvas = document.getElementById('canvas');
        var context = canvas.getContext('2d');
        const video = document.querySelector("#videoElement");

        video.width = 250;
        video.height = 186;

        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({
                video: true
            })
                .then(function (stream) {
                    video.srcObject = stream;
                    //video.play();
                })
                .catch(function (err0r) {
                console.log(err0r)
                console.log("Something went wrong")

                });
        }

        const FPS = 10;

        setInterval(() => {
            //cap.read(src);
            //cv.cvtColor(src, dst, cv.COLOR_BGR2RGB)

            width = video.width
            height = video.height

            context.drawImage(video, 0, 0, width, height);
            
            //cv.imshow(canvas, dst)
            var type = "image/png"
            var mydata = canvas.toDataURL(type);
            data = mydata.replace('data:' + type + ';base64,', ''); //split off junk at the beginning
            context.clearRect(0, 0, width, height);

            socket.emit('image', data);
        }, 10000/FPS);


        socket.on('response_back', function(image){
            const image_id = document.getElementById('image');
            image_id.src = image;
        });
    }