


      var startButton = document.getElementById("startButton");
      var stopButton = document.getElementById("stopButton");


      var socket 

      const video = document.querySelector("#videoElement");

      video.width = 640;
      video.height = 480;

      // video.width = 250;
      // video.height = 186;

      var canvas = document.getElementById("canvas");
      var context = canvas.getContext("2d");

      canvas.width = 640;
      canvas.height = 480;

      canvas.style.display = 'none';
      video.style.display = 'none';

      var image = document.getElementById("image");

      image.width = 640;
      image.height = 480;

      var drawer;

      var emitter;

      function videoLoop() {
          context.drawImage(video, 0, 0, video.width, video.height);
      }


      // if (cv.getBuildInformation)
      // {
      //     console.log(cv.getBuildInformation());
      //     console.log("Loaded CV")
      //     onloadCallback();
      // }
      // else
      // {
      //     // WASM
      //     cv['onRuntimeInitialized']=()=>{
      //         //console.log(cv.getBuildInformation());
      //         onloadCallback();
      //     }
      // }



    
      startButton.onclick = ()=>{
          startButton.disabled = true;

           socket = io()
          //socket.connect('http://' + location.hostname + ':5000/')

          socket.on('connect', function(){
            console.log("Connected...!", socket.connected);
          });

          socket.on('response_back', function(data){
            const arrayBufferView = new Uint8Array(data);
            const blob = new Blob([arrayBufferView], {type: 'image/jpeg'});
            const imageUrl = URL.createObjectURL(blob);
            document.getElementById('image').src = imageUrl;

          });

        

          if (navigator.mediaDevices.getUserMedia) {
              navigator.mediaDevices.getUserMedia({
                  video: true
              })
                  .then(function (stream) {
                      video.srcObject = stream;
                      video.addEventListener('loadeddata', function(){
                      video.play();
                      drawer = setInterval(videoLoop, 1000 / 30);
                    
                    });
                    video.play()
                  })
                  .catch(function (err0r) {
                    console.log(err0r)
                    console.log("Something went wrong")

                  });
          }

              // let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
              // let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
              // let cap = new cv.VideoCapture(video);

            const FPS = 10;

            emiiter = setInterval(() => {
                //cap.read(src);
                //cv.cvtColor(src, dst, cv.COLOR_BGR2RGB)
                //cv.imshow(canvas, dst)
                var type = "image/png"
                var url = canvas.toDataURL(type);
                fetch(url)
                .then(res => res.blob())
                .then(blob =>{
                  if(socket!=null){
                    socket.emit('image', blob)
                  }
                })
            }, 10000/FPS);
            stopButton.disabled = false;

            setTimeout(stopAttendance, 60*1000*3)


      }

      stopButton.onclick = () => {
        stopAttendance()

    }


    function stopAttendance(){
      stopButton.disabled = true;
      if (socket!=null) {
          socket.on('disconnect', function(){
              console.log("Disconnected...", socket.disconnected);
          });
          socket.disconnect();
          if (drawer!=null)
              clearInterval(drawer);
          if (emitter!=null)
              clearInterval(emitter);
      }
      startButton.disabled = false;
      location.href = '/'
    }


