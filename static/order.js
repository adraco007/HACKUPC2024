console.log("Script to select images loaded"); // Debug message
document.getElementById('load_images_button').addEventListener('click', function (event) {
    event.preventDefault(); // Prevent form submission

    let info2 = document.getElementById('info2');
    info2.innerHTML = 'Loading images...'; // Set loading message

    // Get the amount of images from the input field
    let nImages = document.getElementById('imageCountInput').value;

    // Call random selector with the amount of images as a query parameter
    fetch(`/select_images?n_images=${nImages}`)
        .then(response => response.json())
        .then(data => {
            console.log("Response received:", data); // Debug message

            // Update selection bar
            let selectionBar = document.getElementById('selectionBar');
            selectionBar.innerHTML = ''; // Clear previous content
            selectionBar.style.margin = '0'; // Set margin to zero
            selectionBar.style.padding = '0'; // Set padding to zero
            data.imagesToTake.forEach(status => {
                let color = status === 1 ? 'green' : 'red';
                let div = document.createElement('div');
                div.style.width = '1px';
                div.style.height = '5px';
                div.style.backgroundColor = color;
                div.style.display = 'inline-block';
                div.style.margin = '0'; // Remove any margin
                div.style.padding = '0'; // Remove any padding
                selectionBar.appendChild(div);
            });

            let indexs = data.indexs; // Get the indexs of the images
            let similarities = data.similarities; // Get the similarities of the images

            let imageGrid = document.getElementById('imageGrid');
            imageGrid.innerHTML = ''; // Clear previous content

            for (let i = 0; i < indexs.length; i++) {
                let img = document.createElement('img');
                // Use the images_from_index endpoint to get the image
                img.src = `/images_from_index?index=${indexes[i]}`;
                img.style.width = '100px'; // Adjust width as needed
                img.style.height = '100px'; // Adjust height as needed
                img.style.margin = '5px'; // Add some margin between images
                imageGrid.appendChild(img);
            }

            info2.innerHTML = 'These images that belong to different sets are...'; // Set loaded message
        })
        .catch(error => console.error("Fetch error:", error)); // Debug message


});