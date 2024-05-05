// Get buttons prevButton and nextButton
let prevButton = document.getElementById('prevButton');
let nextButton = document.getElementById('nextButton');

// Add event listeners to the buttons
let urls = [];
let actual_url = NaN;

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
                }

                actual_url = urls[0];
                console.log(urls);
                let rightPhoto = document.getElementById('outfitImage');
                rightPhoto.src = actual_url;

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