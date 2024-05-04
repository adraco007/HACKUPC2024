document.getElementById('generatePieceButton').addEventListener('click', function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the description of the ideal cloth from the input field
    let idealClothDescription = document.getElementById('idealClothInput').value;

    // Call the Flask route for generating an image with the provided description
    fetch(`/generate_image?prompt=${idealClothDescription}`)
        .then(response => {
            if (response.ok) {
                // If the response is ok, get the image URL and set it as the src of the image in the outfit HTML
                document.getElementById('outfitImage').src = "/generated_images/image1.png";
                console.log('Image generated successfully');
            } else {
                throw new Error('Failed to generate image');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Handle errors here
        });
});
