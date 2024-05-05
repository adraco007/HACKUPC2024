// Get buttons prevButton and nextButton
let prevButton = document.getElementById('prevButton');
let nextButton = document.getElementById('nextButton');

// Add event listeners to the buttons
let urls = [];
let actual_url = NaN;

// once the person uses the next button:
nextButton.addEventListener('click', function (event) {
    event.preventDefault(); // Prevent form submission
    let index = urls.indexOf(actual_url);
    if (index < urls.length - 1) {
        actual_url = urls[index + 1];
        let rightPhoto = document.getElementById('outfitImage');
        rightPhoto.src = actual_url;
    }
});

// once the person uses the prev button:
prevButton.addEventListener('click', function (event) {
    event.preventDefault(); // Prevent form submission
    let index = urls.indexOf(actual_url);
    if (index > 0) {
        actual_url = urls[index - 1];
        let rightPhoto = document.getElementById('outfitImage');
        rightPhoto.src = actual_url;
    }
});