console.log("Script loaded"); // Debug message
document.getElementById('load_images_button').addEventListener('click', function(event) {   
event.preventDefault(); // Prevent form submission

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
})
.catch(error => console.error("Fetch error:", error)); // Debug message
});