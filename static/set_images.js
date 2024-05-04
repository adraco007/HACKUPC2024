console.log("Script to put images loaded"); // Debug message


document.getElementById('putImagesButton').addEventListener('click', function (event) {   
    event.preventDefault(); // Prevent form submission


    

    /*// Call the Flask function named 'get_images'
    fetch('/get_images')
    .then(response => response.json())
    .then(data => {
        console.log("Response received:", data); // Debug message

        // Update image grid
        let imageGrid = document.getElementById('imageGrid');
        imageGrid.innerHTML = ''; // Clear previous content

        // Create array of urls which follow the format ./data/images/img_0_1.jpg, ./data/images/img_1_1.jpg, etc.
        

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
    .catch(error => console.error("Fetch error:", error)); // Debug message*/
});
