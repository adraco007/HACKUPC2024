// Get buttons prevButton and nextButton
let prevButton = document.getElementById('prevButton');
let nextButton = document.getElementById('nextButton');

let rightPhoto = document.getElementById('outfitImage');

// Take the rectangle that works as the visual indicator
let rectangle = document.getElementById('visual_indicator');

let goToGradio = document.getElementById('goToGradio');

// Add event listeners to the buttons
let urls = [];
let actual_url = NaN;
let urls_to_shop =[NaN, NaN, NaN, NaN, NaN];

document.getElementById('uploadButton').addEventListener('click', function (event) {
    event.preventDefault(); // Prevent form submission

    // Get the selected file
    let fileInput = document.getElementById('fileInput');
    let file = fileInput.files[0];

    if (file) {
        // Create a FormData object to send the file
        let formData = new FormData();
        formData.append('file', file);

        season = document.getElementById('selector1').value;
        product_type = document.getElementById('selector2').value;
        category = document.getElementById('selector3').value;
        formData.append('season', season);
        formData.append('product_type', product_type);
        formData.append('category', category);


        // Call the Flask route for processing the uploaded file
        fetch('/process_uploaded_file', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to process uploaded file');
                }
                // If the response is ok, handle the result
                return response.json();
            })
            .then(data => {
                // Handle the result
                console.log(data);
                indexs = data.indexs;

                for (let i = 0; i < indexs.length; i++) {
                    let index = indexs[i];
                    // Take image from index
                    let url = `/images_from_index?index=${index}`;
                    urls.push(url);

                    // Use get image link to get link from the index of the image
                    fetch('/get_image_link', {
                        method: 'GET',
                        body: JSON.stringify({ index: index})
                    })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error('Failed to get image link');
                            }
                            // If the response is ok, handle the result
                            return response.json();
                        })
                        .then(data => {
                            // Handle the result
                            console.log(data);
                            urls_to_shop.push(data.url);
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            urls_to_shop.push(NaN);
                            // Handle errors here
                        });
                }

                actual_url = urls[0];
                console.log(urls);
                console.log(urls_to_shop);
                rightPhoto.src = actual_url;

                // Check if this url to shop is not NaN, and if so put the rectangle green, else gray
                if (urls_to_shop[0] !== NaN) {
                    rectangle.style.backgroundColor = 'green';
                } else {
                    rectangle.style.backgroundColor = 'gray';
                }

            })
            .catch(error => {
                console.error('Error:', error);
                // Handle errors here
            });
    } else {
        console.error('No file selected');
        // Handle no file selected error
    }
});

document.getElementById('fileInput').addEventListener('change', function () {
    let fileName = this.value.split('\\').pop();
    let label = this.nextElementSibling;
    label.innerText = fileName;

    // Get the selected file
    let fileInput = document.getElementById('fileInput');
    let file = fileInput.files[0];
    // Get the space for the uploaded file
    let leftPhoto = document.getElementById('uploadedImage');
    // Set the uploaded file to the space
    leftPhoto.src = URL.createObjectURL(file);
});


// once the person uses the next button:
nextButton.addEventListener('click', function (event) {
    event.preventDefault(); // Prevent form submission
    let index = urls.indexOf(actual_url);
    if (index < urls.length - 1) {
        actual_url = urls[index + 1];
        let rightPhoto = document.getElementById('outfitImage');
        rightPhoto.src = actual_url;

        // Check if this url to shop is not NaN, and if so put the rectangle green, else gray
        if (urls_to_shop[index + 1] !== NaN) {
            rectangle.style.backgroundColor = 'green';
        } else {
            rectangle.style.backgroundColor = 'gray';
        }
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

        // Check if this url to shop is not NaN, and if so put the rectangle green, else gray
        if (urls_to_shop[index - 1] !== NaN) {
            rectangle.style.backgroundColor = 'red';
        } else {
            rectangle.style.backgroundColor = 'gray';
        }
    }
});

// Add a listener to the right photo to open the link to shop
rightPhoto.addEventListener('click', function () {
    let index = urls.indexOf(actual_url);
    if (urls_to_shop[index] !== NaN) {
        window.open(urls_to_shop[index]);
    }
});

// Add a listener to the rectangle to open the link to goToGradio
goToGradio.addEventListener('click', function () {
    window.open('https://a7d92eb8a3b6a421dc.gradio.live/');
}
);
