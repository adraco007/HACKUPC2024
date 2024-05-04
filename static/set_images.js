console.log("Script to put images loaded"); // Debug message
document.getElementById('putImagesButton').addEventListener('click', function(event) {   
    event.preventDefault(); // Prevent form submission

    // Get the amount of images from the input field
    let nImages = document.getElementById('imageCountInput').value;

    // Call random selector with the amount of images as a query parameter
    fetch(`/select_images?n_images=${nImages}`)
    .then(response => response.json())
    .then(data => {
        console.log("Response received:", data); // Debug message

        // Update image grid
        let imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = ''; // Clear previous content

        // Create grid of images
        for (let i = 0; i < data.imagesToTake.length; i++) {
            let imgUrl = data.imagesToTake[i]; // Assuming each element in data.imagesToTake is an image URL
            let img = document.createElement('img');
            img.src = imgUrl;
            img.style.width = '100px'; // Adjust width as needed
            img.style.height = '100px'; // Adjust height as needed
            img.style.margin = '5px'; // Add some margin between images
            imageGrid.appendChild(img);
        }
    })
    .catch(error => console.error("Fetch error:", error)); // Debug message
});
