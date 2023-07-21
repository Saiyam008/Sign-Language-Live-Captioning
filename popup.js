// const { PythonShell } = require("pytshon-shell");

function predictButtonClick() {
  const videoElement = document.getElementById("screenStreamVideo");

  // Screen sharing code start

  navigator.mediaDevices
    .getDisplayMedia({ video: true, currentWindow: true })
    .then((stream) => {
      videoElement.srcObject = stream;
    })
    .catch((error) => {
      console.error("Error capturing screen: ", error);
    });

  // Screen sharing code end

  // Video mode starts here

  // chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  //   const tab = tabs[0];

  //   chrome.tabCapture.capture(
  //     { audio: false, video: true, tabId: tab.id },
  //     (stream) => {
  //       const videoElement = document.getElementById("screenStreamVideo");
  //       jsonData.module_functions[0].function_name;
  //       videoElement.srcObject = stream;
  //       // for (let key in mappings) {
  //       // const { function_name, arguments } = mappings[key];

  //       // Call Python function using PythonShell

  //       // }
  //     }
  //   );
  // });

  // videoElement.autoplay = true;

  // navigator.mediaDevices
  //   .getUserMedia({ video: true })
  //   .then((stream) => {
  //     console.log(stream);
  //     videoElement.srcObject = stream;
  //     PythonShell.run(
  //       "imageModule.py",
  //       { args: [mainFn, ...arguments.stream] },
  //       function (err, results) {
  //         if (err) throw err;
  //         const predictElementText = document.getElementById("jokeElement");
  //         predictElementText.innerHTML =
  //           `Result of ${function_name}:` + results[0];
  //       }
  //     );
  //   })
  //   .catch((error) => {
  //     console.error("Error accessing camera: ", error);
  //   });

  // For video mode ends here

  // Single picture click mode starts here

  // const imgElement = document.getElementById("capturedImage");

  // navigator.mediaDevices
  //   .getUserMedia({ video: true })
  //   .then((stream) => {
  //     const videoTrack = stream.getVideoTracks()[0];
  //     const imageCapture = new ImageCapture(videoTrack);

  //     // imageCapture.deletePrevImagePromise();

  //     imageCapture
  //       .grabFrame()
  //       .then((imageBitmap) => {
  //         const canvas = document.createElement("canvas");
  //         canvas.width = imageBitmap.width;
  //         canvas.height = imageBitmap.height;
  //         const ctx = canvas.getContext("2d");
  //         ctx.drawImage(imageBitmap, 0, 0);
  //         imgElement.src = canvas.toDataURL("image/png");
  //       })
  //       .catch((error) => {
  //         deletePrevImagePromise();
  //         console.error("Error capturing frame: ", error);
  //       });
  //   })
  //   .catch((error) => {
  //     console.error("Error accessing camera: ", error);
  //   });

  // Single picture click mode ends here
  // }

  // function predictResult() {
  //   const url = "https://icanhazdadjoke.com/slack";
  //   fetch(url)
  //     .then((data) => data.json())
  //     .then((jokeData) => {
  //       const jokeText = jokeData.attachments[0].text;
  //       const jokeElement = document.getElementById("jokeElement");
  //       jokeElement.innerHTML = jokeText;
  //     });
}

// Function to load and predict using the model
async function predictResult() {
  const modelURL = chrome.runtime.getURL("model/model.json");
  const model = (await loadLayersModel(modelURL)).execute;
}

document.addEventListener("DOMContentLoaded", function () {
  const predictButton = document.getElementById("predictButton");
  predictButton.addEventListener("click", predictButtonClick);
  predictButton.addEventListener("click", predictResult);
});
