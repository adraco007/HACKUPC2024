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

                urls = data.urls;
                actual_url = urls[0];
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


