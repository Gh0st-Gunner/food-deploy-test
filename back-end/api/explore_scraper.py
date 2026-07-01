import requests
from bs4 import BeautifulSoup
import json
import re
import time
from typing import List, Dict

# In-memory caches by macro signature
# Format: {cache_key: {"timestamp": float, "items": list}}
EXPLORE_CACHES: Dict[str, Dict] = {}
CACHE_TTL = 3600  # 1 hour in seconds

DEFAULT_DISHES = [
    {
        "title": "Lemongrass Chicken Noodle Bowls (Bún Gà Nướng)",
        "link": "https://www.skinnytaste.com/lemongrass-chicken-noodle-bowls/",
        "description": "Inspired by Vietnamese bún gà nướng—grilled lemongrass-marinated chicken thighs served over rice vermicelli noodles, fresh herbs, and crisp vegetables.",
        "image_url": "https://images.unsplash.com/photo-1598515214211-89d3e73ae83b?auto=format&fit=crop&q=80&w=400",
        "calories": 541,
        "protein": 43,
        "carbs": 57,
        "fat": 15,
        "recipe_ingredients": [
            "1.5 lbs boneless skinless chicken thighs, trimmed of fat",
            "1.5 tablespoons lemongrass paste or grated fresh lemongrass",
            "3 garlic cloves, minced",
            "2 tablespoons fish sauce",
            "1 tablespoon low-sodium soy sauce",
            "1 tablespoon sugar or honey",
            "6 oz dried vermicelli rice noodles",
            "2 cups bean sprouts, 2 Persian cucumbers, julienned",
            "Fresh mint, cilantro, and sliced carrots",
            "1/4 cup roasted peanuts, crushed"
        ],
        "recipe_instructions": [
            "Combine lemongrass paste, garlic, fish sauce, soy sauce, and sugar to marinate the chicken for at least 2 hours.",
            "Make Nuoc Cham by mixing fish sauce, lime juice, sweetener, and warm water in a small bowl.",
            "Sear the chicken thighs in a pan over medium heat for about 5 minutes on each side until cooked through.",
            "Soak the vermicelli noodles in boiling water for 3 minutes, then drain and rinse with cold water.",
            "Assemble bowls with noodles, fresh sliced veggies, fresh herbs, sliced chicken, crushed peanuts, and drizzle with Nuoc Cham dressing."
        ]
    },
    {
        "title": "Vietnamese Vermicelli Noodle Salad with Grilled Pork",
        "link": "https://primetasty.com/vietnamese-vermicelli-noodle-salad-grilled-pork/",
        "description": "Tender grilled pork loin slices served over a refreshing bed of fresh vegetables, aromatic herbs, and vermicelli rice noodles.",
        "image_url": "https://images.unsplash.com/photo-1544025162-d76694265947?auto=format&fit=crop&q=80&w=400",
        "calories": 450,
        "protein": 25,
        "carbs": 50,
        "fat": 15,
        "recipe_ingredients": [
            "300g pork loin, thinly sliced",
            "200g rice vermicelli noodles",
            "2 tablespoons fish sauce",
            "1 tablespoon sugar, 1 tablespoon soy sauce, 1 tablespoon vegetable oil",
            "1 clove garlic, minced",
            "1 carrot and 1 cucumber, julienned",
            "Fresh mint, cilantro, and lettuce leaves",
            "Crushed peanuts and lime wedges for serving"
        ],
        "recipe_instructions": [
            "Mix fish sauce, sugar, soy sauce, oil, and minced garlic to create a marinade.",
            "Marinate the thinly sliced pork loin for at least 30 minutes in the refrigerator.",
            "Soak rice vermicelli noodles in hot water for 15 minutes until soft, then drain.",
            "Preheat a grill or grill pan over medium-high heat. Grill the marinated pork for 3-5 minutes on each side.",
            "Assemble the noodle salad: layer lettuce, vermicelli noodles, grilled pork, carrots, cucumbers, mint, and cilantro.",
            "Garnish with crushed peanuts and serve with fresh lime wedges."
        ]
    },
    {
        "title": "Vietnamese Chicken Rice Paper Rolls",
        "link": "https://www.weareathleats.com/recipe/vietnamese-chicken-rice-paper-rolls-2",
        "description": "Fresh, light rice paper rolls loaded with shredded chicken breast, creamy avocado, crisp red bell pepper, and fresh cucumber slices.",
        "image_url": "https://images.unsplash.com/photo-1534422298391-e4f8c172dddb?auto=format&fit=crop&q=80&w=400",
        "calories": 297,
        "protein": 27,
        "carbs": 27,
        "fat": 9,
        "recipe_ingredients": [
            "2 Vietnamese rice paper sheets",
            "75g rotisserie chicken breast, shredded",
            "0.5 teaspoon fresh lime juice",
            "0.25 avocado, thinly sliced",
            "0.25 red bell pepper, julienned",
            "0.25 cucumber, julienned",
            "Handful of fresh shredded lettuce and basil leaves",
            "2 tablespoons soy sauce, 0.5 teaspoon rice vinegar, 1 teaspoon honey"
        ],
        "recipe_instructions": [
            "Toss the shredded chicken with lime juice and a dash of soy sauce.",
            "Prepare the dipping sauce by whisking soy sauce, rice vinegar, honey, and chili flakes.",
            "Dip a rice paper sheet in lukewarm water until soft, then lay flat on a plate.",
            "Layer the chicken, avocado, bell pepper, cucumber, lettuce, and basil leaves in the center.",
            "Fold in the sides of the paper and roll up tightly.",
            "Serve immediately with the dipping sauce."
        ]
    },
    {
        "title": "Bún Thịt Nướng Chả Giò (Vietnamese Noodle Bowls)",
        "link": "https://www.foodfaithfitness.com/vietnamese-noodles/",
        "description": "Experience Vietnamese street food: a colorful mix of grilled lemongrass pork shoulder and crispy spring rolls over vermicelli noodles.",
        "image_url": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&q=80&w=400",
        "calories": 453,
        "protein": 22,
        "carbs": 69,
        "fat": 10,
        "recipe_ingredients": [
            "1 lb pork shoulder, thinly sliced",
            "2 tbsp soy sauce, 1 tbsp fish sauce, 2 tbsp brown sugar",
            "2 garlic cloves, minced, and 1 stalk lemongrass, chopped",
            "8 oz rice vermicelli noodles",
            "4 Vietnamese spring rolls, cooked and sliced",
            "1 head lettuce, shredded",
            "1 carrot and 1 cucumber, julienned",
            "1/4 cup fresh mint and cilantro, 1/4 cup roasted peanuts, crushed"
        ],
        "recipe_instructions": [
            "Marinate pork shoulder slices in soy sauce, fish sauce, brown sugar, garlic, and lemongrass for 30 minutes.",
            "Soak vermicelli noodles in hot water for 10 minutes, drain and rinse with cold water.",
            "Grill the marinated pork slices for 3-4 minutes per side until fully cooked and slightly charred.",
            "Mix fish sauce, sugar, lime juice, water, minced garlic, and chili for the Nuoc Cham sauce.",
            "Assemble: place noodles in bowls, top with pork, sliced spring rolls, lettuce, carrots, and cucumber. Garnish with peanuts and serve with dressing."
        ]
    },
    {
        "title": "Vietnamese Lemongrass Vermicelli Bowls (Bun Ga Nuong)",
        "link": "https://www.feastingathome.com/vietnamese-vermicelli-bowl/",
        "description": "Fresh and vibrant vermicelli bowls topped with grilled lemongrass-marinated chicken breast or tofu, crisp vegetables, and herbs.",
        "image_url": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?auto=format&fit=crop&q=80&w=400",
        "calories": 564,
        "protein": 14,
        "carbs": 60,
        "fat": 16,
        "recipe_ingredients": [
            "1.2 lbs chicken breast (or tofu), sliced",
            "1/2 cup shallots, chopped, and 2 garlic cloves",
            "1/2 cup finely chopped white part of lemongrass",
            "1 tsp Chinese five spice, 1 tsp salt, 1 tsp sugar, 1/4 cup oil",
            "8 oz vermicelli rice noodles, soaked and drained",
            "1 red bell pepper, sliced, and 1 cucumber, sliced",
            "Fresh Thai basil, mint, and cilantro",
            "Dressing: 1/4 cup lime juice, 1 tbsp fish sauce, 2 tbsp water, 1 tbsp honey"
        ],
        "recipe_instructions": [
            "Pulse lemongrass, shallots, garlic, salt, sugar, five-spice, and oil to create a marinade paste.",
            "Coat chicken or tofu slices in the lemongrass paste and marinate for 30 minutes.",
            "Cook vermicelli noodles in boiling water for 3 minutes, rinse, and set aside.",
            "Sear the protein in a hot skillet for 5-7 minutes per side until golden brown.",
            "Assemble: arrange noodles, fresh vegetables, fresh herbs, and sliced chicken/tofu in a wide bowl. Spoon dressing over and serve."
        ]
    }
]

def parse_recipe_from_text(content: str) -> Dict:
    # Default fallbacks
    default_img = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&q=80&w=400"
    calories = 400
    protein = 18
    carbs = 50
    fat = 12
    
    ingredients = [
        "Fresh local Vietnamese herbs",
        "Rice noodles or wrapper sheets",
        "Proteins (shrimp, pork, beef or chicken)",
        "Traditional sweet chili dipping sauce"
    ]
    instructions = [
        "Prep all fresh ingredients, washing and slicing herbs carefully.",
        "Boil vermicelli or cook proteins (seasoned with garlic and fish sauce).",
        "Assemble the dish elegantly and garnish with raw greens.",
        "Drizzle with fish sauce dressing or dipping sauce and serve immediately."
    ]
    
    try:
        # Parse nutrients
        cal_match = re.search(r'(?:Calories|Cals?|Energy):\s*(\d+)', content, re.IGNORECASE)
        if cal_match:
            calories = int(cal_match.group(1))
        else:
            cal_match2 = re.search(r'(\d+)\s*(?:kcal|calories)', content, re.IGNORECASE)
            if cal_match2:
                calories = int(cal_match2.group(1))
                
        prot_match = re.search(r'(?:Protein):\s*(\d+)', content, re.IGNORECASE)
        if prot_match:
            protein = int(prot_match.group(1))
        else:
            prot_match2 = re.search(r'(\d+)g\s*protein', content, re.IGNORECASE)
            if prot_match2:
                protein = int(prot_match2.group(1))
                
        carbs_match = re.search(r'(?:Carbohydrates|Carbs?):\s*([\d\.]+)', content, re.IGNORECASE)
        if carbs_match:
            carbs = int(float(carbs_match.group(1)))
        else:
            carbs_match2 = re.search(r'([\d\.]+)g\s*carbs?', content, re.IGNORECASE)
            if carbs_match2:
                carbs = int(float(carbs_match2.group(1)))
                
        fat_match = re.search(r'(?:Total\s+)?Fats?:\s*([\d\.]+)', content, re.IGNORECASE)
        if fat_match:
            fat = int(float(fat_match.group(1)))
        else:
            fat_match2 = re.search(r'([\d\.]+)g\s*fats?', content, re.IGNORECASE)
            if fat_match2:
                fat = int(float(fat_match2.group(1)))

        # Parse ingredients
        ing_header_match = re.search(r'###\s*Ingredients|##\s*Ingredients|Ingredients\s*:', content, re.IGNORECASE)
        if ing_header_match:
            start_idx = ing_header_match.end()
            end_idx = content.find('###', start_idx)
            if end_idx == -1:
                end_idx = content.find('##', start_idx)
            if end_idx == -1:
                end_idx = len(content)
                
            ing_chunk = content[start_idx:end_idx]
            parsed_ings = []
            for line in ing_chunk.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or (line and line[0].isdigit() and 'g' in line):
                    item = re.sub(r'^[\-\*\s]+', '', line).strip()
                    if item and len(item) < 100:
                        parsed_ings.append(item)
            if len(parsed_ings) >= 3:
                ingredients = parsed_ings[:10]

        # Parse instructions
        ins_header_match = re.search(r'###\s*Instructions|##\s*Instructions|Instructions\s*:', content, re.IGNORECASE)
        if ins_header_match:
            start_idx = ins_header_match.end()
            end_idx = content.find('###', start_idx)
            if end_idx == -1:
                end_idx = content.find('##', start_idx)
            if end_idx == -1:
                end_idx = len(content)
                
            ins_chunk = content[start_idx:end_idx]
            parsed_ins = []
            for line in ins_chunk.split('\n'):
                line = line.strip()
                if re.match(r'^\d+[\.\)\s]+', line):
                    step = re.sub(r'^\d+[\.\)\s]+', '', line).strip()
                    if step:
                        parsed_ins.append(step)
            if len(parsed_ins) >= 2:
                instructions = parsed_ins[:8]
    except Exception:
        pass

    # Image mapping
    img_mapped = default_img
    lower_content = content.lower()
    if 'chicken' in lower_content or 'ga' in lower_content:
        img_mapped = "https://images.unsplash.com/photo-1598515214211-89d3e73ae83b?auto=format&fit=crop&q=80&w=400"
    elif 'pork' in lower_content or 'thịt nướng' in lower_content:
        img_mapped = "https://images.unsplash.com/photo-1544025162-d76694265947?auto=format&fit=crop&q=80&w=400"
    elif 'roll' in lower_content or 'cuốn' in lower_content:
        img_mapped = "https://images.unsplash.com/photo-1534422298391-e4f8c172dddb?auto=format&fit=crop&q=80&w=400"
    elif 'shrimp' in lower_content or 'tom' in lower_content:
        img_mapped = "https://images.unsplash.com/photo-1565557623262-b51c2513a641?auto=format&fit=crop&q=80&w=400"

    return {
        "image_url": img_mapped,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "recipe_ingredients": ingredients,
        "recipe_instructions": instructions
    }

def scrape_ollama_search_feed(
    calories: int = None,
    protein: int = None,
    carbs: int = None,
    fat: int = None,
    ingredients: str = None
) -> List[Dict]:
    from core.settings import get_settings
    settings = get_settings()

    url = "https://ollama.com/api/web_search"
    token = settings.ollama_token or "b03338bd9ac347d38847a7bcd80f5e0f.37ZKaVtFf7B7CHNv0HxD2H8P"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Dynamically build query fit for the input BMR goals and available ingredients
    if ingredients:
        query = f"healthy Vietnamese recipes using {ingredients} from sites like cookpad.com, dienmayxanh.com, hungryhuy.com, runawayrice.com, vickypham.com"
        if calories or protein or carbs or fat:
            parts = []
            if calories:
                parts.append(f"around {calories} kcal")
            if protein:
                parts.append(f"around {protein}g protein")
            if carbs:
                parts.append(f"around {carbs}g carbs")
            if fat:
                parts.append(f"around {fat}g fat")
            query += f" with {', '.join(parts)}"
        query += " with ingredients and instructions"
    elif calories or protein or carbs or fat:
        parts = []
        if calories:
            parts.append(f"around {calories} kcal")
        if protein:
            parts.append(f"around {protein}g protein")
        if carbs:
            parts.append(f"around {carbs}g carbs")
        if fat:
            parts.append(f"around {fat}g fat")
        query = f"healthy Vietnamese recipes with {', '.join(parts)} from cookpad.com, dienmayxanh.com, hungryhuy.com, runawayrice.com, vickypham.com with ingredients instructions"
    else:
        query = "healthy Vietnamese recipes from cookpad.com, dienmayxanh.com, hungryhuy.com, runawayrice.com, vickypham.com with calories protein carbs fat ingredients instructions"

    data = {
        "query": query,
        "max_results": 10
    }
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=8)
        r.raise_for_status()
        res_json = r.json()
        results = res_json.get("results", [])
        
        items = []
        for item in results:
            title = item.get("title", "")
            link = item.get("url", "")
            content = item.get("content", "")
            
            # Create a clean description
            desc_clean = content.split('\n')[0]
            if len(desc_clean) > 130:
                desc_clean = desc_clean[:127] + "..."
                
            items.append({
                "title": title,
                "link": link,
                "description": desc_clean or "Delicious traditional Vietnamese food recipe.",
                "content": content
            })
        return items
    except Exception as e:
        print(f"Error calling Ollama search API: {e}")
        return []

def get_explore_dishes(
    calories: int = None,
    protein: int = None,
    carbs: int = None,
    fat: int = None
) -> List[Dict]:
    """Retrieves and returns explore dishes list with in-memory TTL caching based on macro goals."""
    global EXPLORE_CACHES
    
    cache_key = f"{calories or ''}-{protein or ''}-{carbs or ''}-{fat or ''}"
    current_time = time.time()
    
    if cache_key in EXPLORE_CACHES:
        cache = EXPLORE_CACHES[cache_key]
        if cache["items"] and (current_time - cache["timestamp"] < CACHE_TTL):
            return cache["items"]
        
    print(f"Scraping fresh explore dishes using Ollama Web Search API for {cache_key}...")
    feed_items = scrape_ollama_search_feed(
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat
    )
    
    if not feed_items:
        print("Ollama Web Search API failed. Returning default seeded recipes.")
        return DEFAULT_DISHES
        
    scraped_dishes = []
    for item in feed_items:
        details = parse_recipe_from_text(item["content"])
        scraped_dishes.append({
            "title": item["title"],
            "link": item["link"],
            "description": item["description"],
            **details
        })
        
    # If the list is empty, default it
    if not scraped_dishes:
        scraped_dishes = DEFAULT_DISHES
        
    EXPLORE_CACHES[cache_key] = {
        "timestamp": current_time,
        "items": scraped_dishes
    }
    return scraped_dishes

def generate_fallback_recipe(ingredients: str, calories: int = None, protein: int = None, carbs: int = None, fat: int = None) -> List[Dict]:
    """Generates a mock recipe based on the input ingredients when Ollama Search is offline."""
    import re
    # Clean and split ingredients
    ing_list = [i.strip() for i in re.split(r'[,\s]+', ingredients) if i.strip()]
    if not ing_list:
        ing_list = ["healthy vegetables"]
        
    title = f"Vietnamese Stir-Fry with " + " and ".join([i.capitalize() for i in ing_list[:2]])
    if len(ing_list) > 2:
        title += " & Herbs"
        
    # Estimate reasonable macros based on target calories
    cal_target = calories or 400
    p_target = protein or int((cal_target * 0.25) / 4)
    c_target = carbs or int((cal_target * 0.45) / 4)
    f_target = fat or int((cal_target * 0.30) / 9)
    
    # Construct ingredients list
    recipe_ingredients = [f"200g of fresh {ing}" for ing in ing_list]
    recipe_ingredients.extend([
        "2 cloves garlic, minced",
        "1 tablespoon vegetable oil",
        "1.5 tablespoons premium Vietnamese fish sauce (nước mắm)",
        "1 teaspoon sugar",
        "Fresh cilantro and chopped green onions for garnish"
    ])
    
    # Construct instructions
    recipe_instructions = [
        "Wash and prep all ingredients, slicing them into bite-sized pieces.",
        "Heat the vegetable oil in a pan or wok over medium-high heat.",
        "Add the minced garlic and sauté for 1 minute until fragrant.",
        f"Add the {', '.join(ing_list)} and stir-fry for 5-7 minutes until tender.",
        "Drizzle the fish sauce and sprinkle sugar over the ingredients, tossing to coat evenly.",
        "Garnish with fresh cilantro and chopped green onions. Serve hot."
    ]
    
    # Return structured recipe
    return [{
        "title": title,
        "link": "https://healthyvietnameserecipes.com/custom-stir-fry",
        "description": f"A quick, healthy Vietnamese style stir-fry featuring fresh {', '.join(ing_list)}. High in nutrition and tailored to your macro goals.",
        "image_url": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&q=80&w=400",
        "calories": cal_target,
        "protein": p_target,
        "carbs": c_target,
        "fat": f_target,
        "recipe_ingredients": recipe_ingredients,
        "recipe_instructions": recipe_instructions
    }]

def generate_recipes_from_ingredients(
    ingredients: str,
    calories: int = None,
    protein: int = None,
    carbs: int = None,
    fat: int = None
) -> List[Dict]:
    """Scrapes recipes containing the given ingredients and fitting optional target macros."""
    print(f"Scraping recipes for ingredients: {ingredients}...")
    feed_items = scrape_ollama_search_feed(
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        ingredients=ingredients
    )
    
    scraped_dishes = []
    if feed_items:
        for item in feed_items:
            details = parse_recipe_from_text(item["content"])
            scraped_dishes.append({
                "title": item["title"],
                "link": item["link"],
                "description": item["description"],
                **details
            })
            
    # Fallback to local generator if Ollama Search is offline or returned empty list
    if not scraped_dishes:
        print("Ollama Web Search API failed or returned empty results. Returning dynamic backup recipe.")
        scraped_dishes = generate_fallback_recipe(ingredients, calories, protein, carbs, fat)
        
    return scraped_dishes
