document
.getElementById("selectAllButton")
.addEventListener("click", function () {
  const likeIcons = document.querySelectorAll(".like-icon");

  likeIcons.forEach(function (icon) {
    icon.classList.add("fas");
    icon.classList.add("far");

    const imageContainer = icon.parentElement;
    imageContainer.classList.add("liked");
    var imageNumber = imageContainer
      .querySelector("img")
      .alt.split("/")[1];
    imageNumber = imageNumber.split(".")[0];

    likedImages.push(Number(imageNumber));

    if (likedImages.includes(Number(imageNumber))) {
      icon.classList.add("fas");
      icon.classList.add("far");
    }
    if (likedImages.length >= 5) {
      document.getElementById("executeButton").disabled = false;
    } else {
      document.getElementById("executeButton").disabled = true;
    }
  });
});
const deleteButton = document.getElementById("resetButton");

deleteButton.addEventListener("click", function () {
original_array = [];
content_array = [];
hybrid_array = [];
collab_array = [];
likedImages = [];
mode = "rand";
for (let i = rand_array.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [rand_array[i], rand_array[j]] = [rand_array[j], rand_array[i]];
}
updateGallery();
document.getElementById("executeButton").disabled = true;
document.getElementById("dropdownMenuButton").innerText =
  "Content-based";
});
const folder = "base/";
var original_array = [];
var content_array = [];
var hybrid_array = [];
var collab_array = [];
var mode = "rand";
var likedImages = [];

document
.getElementById("executeButton")
.addEventListener("click", function () {
  document.getElementById("loadingModal").style.display = "block";
  document.getElementById("loadingOverlay").style.display = "block";
  executeFunction();
});

function executeFunction() {
switch (mode) {
  case "content":
    if (content_array) break;
  case "original":
    break;
  case "hybrid":
    break;
  case "collab":
    break;

  case "rand":
    mode = "content";
    break;
}
const data = JSON.stringify({ likedImages: likedImages, mode: mode });
fetch("/getImages", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: data,
})
  .then((response) => response.json())
  .then((data) => {
    ///console.log(data);
    switch (mode) {
      case "content":
        content_array = data.images;
        break;
      case "hybrid":
        hybrid_array = data.images;
        break;
      case "collab":
        collab_array = data.images;
        break;
    }
    original_array = data.original;
    document.getElementById("hiddenDiv").removeAttribute("hidden");
    document.getElementById("loadingModal").style.display = "none";
    document.getElementById("loadingOverlay").style.display = "none";

    updateGallery();
  })
  .catch((error) => console.error("Error:", error));
}
document.querySelectorAll(".dropdown-item").forEach((item) => {
item.addEventListener("click", function () {
  document.getElementById("dropdownMenuButton").innerText =
    this.innerText;
  switch (document.getElementById("dropdownMenuButton").innerText) {
    case "Content-based":
      mode = "content";
      break;
    case "Liked":
      mode = "original";
      break;
    case "Hybrid":
      mode = "hybrid";
      break;
    case "Collaborative":
      mode = "collab";
      break;
    default:
      mode = "rand";
      break;
  }
  updateGallery();
});
});

var n = 516;
var gallery = document.getElementById("gallery");

let rand_array = [];
for (let i = 0; i < n; i++) {
rand_array.push(i);
}

for (let i = rand_array.length - 1; i > 0; i--) {
const j = Math.floor(Math.random() * (i + 1));
[rand_array[i], rand_array[j]] = [rand_array[j], rand_array[i]];
}

function updateGallery() {
var length = 100;
///console.log(mode);
gallery.innerHTML = "";
var array;
switch (mode) {
  case "content":
    array = content_array;
    break;
  case "original":
    array = original_array;
    break;
  case "hybrid":
    array = hybrid_array;
    break;
  case "collab":
    array = collab_array;
    break;
  case "rand":
    array = rand_array;
    break;
}
if (mode != "rand") length = array.length;

///console.log(array);
for (let i = 0; i < length; i++) {
  const container = document.createElement("div");
  container.classList.add("image-container");
  const image = document.createElement("img");

  image.src = folder + (array[i] + 1) + ".jpg";
  image.alt = "Image:" + folder + (array[i] + 1) + ".jpg";
  image.classList.add("image");

  const likeIcon = document.createElement("i");
  likeIcon.classList.add("like-icon", "far", "fa-heart");
  container.appendChild(image);
  container.appendChild(likeIcon);
  image.addEventListener("click", function () {
    document.getElementById("modalImage").src = this.src;
    $("#imageModal").modal("show");
  });
  gallery.appendChild(container);
}
const likeIcons = document.querySelectorAll(".like-icon");

likeIcons.forEach(function (icon) {
  icon.addEventListener("click", function () {
    icon.classList.toggle("fas");
    icon.classList.toggle("far");

    const imageContainer = icon.parentElement;
    imageContainer.classList.toggle("liked");
    var imageNumber = imageContainer
      .querySelector("img")
      .alt.split("/")[1];
    imageNumber = imageNumber.split(".")[0];
    let index = Number(likedImages.indexOf(Number(imageNumber)));
    if (index != -1) {
      likedImages.splice(index, 1);
    } else {
      likedImages.push(Number(imageNumber));
    }
    ///console.log(likedImages);
    if (likedImages.length >= 5) {
      document.getElementById("executeButton").disabled = false;
    } else {
      document.getElementById("executeButton").disabled = true;
    }
  });
  if (
    likedImages.includes(
      Number(
        icon.parentElement
          .querySelector("img")
          .alt.split("/")[1]
          .split(".")[0]
      )
    )
  ) {
    icon.classList.toggle("fas");
    icon.classList.toggle("far");
    icon.parentElement.classList.toggle("liked");
  }
});
}
updateGallery();