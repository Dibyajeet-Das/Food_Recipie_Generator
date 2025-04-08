
/*--------------------------------------------------------------------------------------*/
//Function is used for menu toggle when the screen is small
document.addEventListener("DOMContentLoaded", function () {
    //DOMContentLoaded helps to make the html elements available for any changes
    const hamburger = document.getElementById("hamburger");
    const navMenu = document.getElementById("nav-menu");

    hamburger.addEventListener("click", () => {
        navMenu.classList.toggle("show");
    });
});
/*--------------------------------------------------------------------------------------*/
let images = ['/static/images/1.jpg', '/static/images/2.jpg', '/static/images/3.jpg'];
let index = 0;

function changeBackground() {
    document.querySelector('.hero').style.backgroundImage = `url(${images[index]})`;
    index = (index + 1) % images.length;
}
setInterval(changeBackground, 3000);

/*Adding the functionality where the cards get changed after a period of time*/
//know we will declare the cards where the each card is present with front and back element
//for smoth transition
const FoodCards = [
{
    FrontCard:['/static/images/Dish1.jpg','/static/images/Dish2.jpg','/static/images/Dish3.jpg'],
    BackCard:['An "egg chicken roll" is a street food dish, particularly popular in Kolkata, India, where a paratha (flatbread) is filled with stir-fried chicken and scrambled egg, then rolled up'
    ,'Probably the best-known Korean dish, bibimbap originated on the eve of Lunar New Year when it was traditional to use up all the vegetables and side-dishes in the house. A hot stone bowl is filled with cooked rice and topped with vegetables, pickled Chinese radish, carrot and mushrooms.'
    ,'Biryani is a flavorful, layered dish of rice, meat or seafood, and spices that originated in South Asia. Its commonly made with chicken, lamb, or mutton, but can also be made with beef, prawns, or fish. '],
},
{
    FrontCard:['/static/images/Dish4.jpg','/static/images/Dish5.jpg','/static/images/Dish6.jpg'],
    BackCard:['This rice bowl dish is almost as popular as ramen in Japan and a common lunchtime choice among busy Japanese workers. Donburi is made by preparing (normally by simmering or frying) various meat, fish and vegetables and serving over steamed rice in large bowls.'
    ,'chicken fried steak was born to go with American food classics like mashed potatoes and black-eyed peas.A slab of tenderized steak breaded in seasoned flour and pan fried, it’s kin to the Weiner Schnitzel brought to Texas by Austrian and German immigrants, who adapted their veal recipe to use the bountiful beef found in Texas.'
    ,'The most frequently preferred pizza recipe in Italy is Margarita, and it is consumed all over the world with its light structure. Prepared in a very practical way, Margarita contains flour, olive oil, water, dry yeast, salt, granulated sugar, tomatoes, oregano, basil, and mozzarella cheese.'],
},
{
    FrontCard:['/static/images/Dish7.jpg','/static/images/Dish8.jpg','/static/images/Dish9.jpg'],
    BackCard:['Shchi is a deceptively simple soup with a complex taste. What may look like a simple cabbage soup is actually a filling but light soup made from sauerkraut, cabbage, or other green leaves. Shchi is an integral part of Russian cuisine, and has been eaten almost daily for centuries in Russia'
    ,'DA famous Indo-Chinese dish that combines tender chicken pieces with flavorful noodles. The savoury broth enhances the taste and complements the textures of the dish, making it a popular choice for both quick meals and special occasions.'
    ,'Jjigae is a type of rich, spicy stew. This seafood and silken tofu version is called sundubu and is served like bibimbap in a hot stone bowl. Discover more warming stew recipes with our winter stew recipes and vegan stew recipes.'],
},
];

//Defining the index
let CurrentIndex = 0;

//Here we are selecting all the cards
const AllFoodCards = document.querySelectorAll('.image-box');

//Creating a function to update the cards with the single function
function ChangeCards() {
    
    //querySelector used to precisely identify and interact with specific elements on a web page
    AllFoodCards.forEach((card,index) => {
        const FrontImage = card.querySelector('.image-box-front img'); 
        const BackImage = card.querySelector('.image-box-back');

        // Remove all animation classes first to restart the animation
        card.classList.remove('slide-out', 'slide-in', 'active');

        // Force a small delay to allow reflow and re-trigger animation
        void card.offsetWidth; // This line forces the browser to reflow

        // Start slide-out animation
        card.classList.add('slide-out');

        setTimeout(() => {
            // Update images and descriptions
            FrontImage.src = FoodCards[CurrentIndex].FrontCard[index];
            FrontImage.alt = `Image ${CurrentIndex + 1} - ${index + 1}`;
            BackImage.innerHTML = `<p>${FoodCards[CurrentIndex].BackCard[index]}</p>`;

            // Reset animation classes
            card.classList.remove('slide-out');
            card.classList.add('slide-in');

            setTimeout(() => {
                card.classList.remove('slide-in');
                card.classList.add('active');
            }, 400); // Small delay to smooth out transition
        }, 900); // Matches CSS transition duration
});

// Explicitly reset or increment CurrentIndex
if (CurrentIndex >= FoodCards.length - 1) {
CurrentIndex = 0;
} else {
CurrentIndex++;
}
}
//Know we need to Update Our Card in every 5 secound
setInterval(ChangeCards,9000);

/*--------------------------------------------------------------------------------------*/

const text = ['"Samosa"','"Kebab"','"Chicken tikka"','"Tacos"','"Burger"'];
let n = 0;

function ChangePlaceHolder() {
    const SearchBox = document.getElementById('recipe-search');
    SearchBox.setAttribute("placeholder", `Search for recipes like : ${text[index]}`);
    n = (n + 1) % text.length; // Loop back to the first item
}

setInterval(ChangePlaceHolder,2000);


/*--------------------------------------------------------------------------------------*/
document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM Loaded");

    const searchButton = document.getElementById("search-btn");
    const resultDiv = document.getElementById("recipe-results");

    if (!searchButton || !resultDiv) {
        console.error("❌Error: Button or result div not found.");
        return;
    }

    searchButton.addEventListener("click", async function () {
        let inputField = document.getElementById("recipe-search");
        //Here .value helps to extract the text and trim() function helps to remove the extra spaces
        //from the start and end of the string
        let dish = inputField.value.trim();

        if (!dish) {
            resultDiv.innerHTML = "<p style='color:red;'>⚠ Please enter a dish name.</p>";
            return;
        }

        resultDiv.innerHTML = `
        <div class="loading-container">
            <p>🔄 Generating recipe...</p>
            <img src="/static/images/Generate.gif" alt="Loading" class="loading-gif">
        </div>
        `;

        try {
            console.log(`🔍 Fetching: /get-recipe/?dish=${dish}`);
            //await is used to wait for the fetch function to complete before moving
            //encodes special characters in a string so it can safely be used in a URL.helps change the the spaces into symbols
            //any special character in the string which will easy to transmit over the network
            let response = await fetch(`/get-recipe/?dish=${encodeURIComponent(dish)}`);

            if (!response.ok) {
                let errorData = await response.json();
                throw new Error(errorData.error || "Failed to fetch recipe.");
            }

            let data = await response.json();
            console.log("✅ Recipe Data:", data);

            // Convert recipe text into points
            let recipeLines = data.recipe.split("\n").filter(line => line.trim() !== "");  
            let formattedRecipe = recipeLines.map(line => `<li>${line}</li>`).join("");

            resultDiv.innerHTML = `
                <div class="recipe-box">
                    <h3>${data.dish}</h3>
                    <ul>${formattedRecipe}</ul>
                </div>
            `;
        } catch (error) {
            console.error("Error fetching recipe:", error);
            resultDiv.innerHTML = "<p style='color:red;'>❌ Failed to fetch recipe. Try again.</p>";
        }
    });
});


document.querySelector('.subscribe-form').addEventListener('submit', function (e) {
    e.preventDefault(); // this  function helps to prevent from reloading the page
    const email = this.querySelector('input').value;
    alert("Thanks for subscribing, " + email );
    this.reset(); // after the user compilation of information reset the form
  });
 
  