function predictButtonClick() {
  const videoElement = document.getElementById("screenStreamVideo");

  // 
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];

    chrome.tabCapture.capture(
      { audio: false, video: true, tabId: tab.id },
      (stream) => {
        const videoElement = document.getElementById("screenStreamVideo");
        videoElement.srcObject = stream;
      }
    );
  });

  videoElement.autoplay = true;

  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      console.log(stream);
      videoElement.srcObject = stream;
    })
    .catch((error) => {
      console.error("Error accessing camera: ", error);
    });


  // Make an AJAX request to the Python server
  fetch("/predict")
    .then((response) => response.json())
    .then((result) => {
      // Update the HTML with the prediction result
      const jokeElement = document.getElementById("jokeElement");
      jokeElement.innerHTML = result.prediction;
    })
    .catch((error) => {
      console.error("Error predicting: ", error);
    });
}

document.addEventListener("DOMContentLoaded", function () {
  const predictButton = document.getElementById("predictButton");
  predictButton.addEventListener("click", predictButtonClick);
});
