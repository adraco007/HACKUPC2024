document.getElementById('generatePieceButton').addEventListener('click', function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the description of the ideal cloth from the input field
    let idealClothDescription = document.getElementById('idealClothInput').value;

    // Call the Flask route for generating an image with the provided description
    fetch(`/generate_image?prompt=${idealClothDescription}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate image');
            }
            // If the response is ok, call the route to get the generated image
            return fetch('/get_generated_image');
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to get generated image');
            }
            // If the response is ok, get the image URL and set it as the src of the image in the outfit HTML
            return response.json();
        })
        .then(data => {
            let imageUrl = data.imageUrl; // Assuming the response JSON contains the image URL
            // Set the src attribute of the image in the outfit HTML
            document.getElementById('outfitImage').src = imageUrl;
        })
        .catch(error => {
            console.error('Error:', error);
            // Handle errors here
        });
});
