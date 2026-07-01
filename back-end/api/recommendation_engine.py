import math
import re
from typing import List, Dict

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(a * a for a in v1))
    m2 = math.sqrt(sum(a * a for a in v2))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)

def recommend_dishes(
    user_profile: Dict,
    recent_meals: List[Dict],
    candidate_dishes: List[Dict]
) -> List[Dict]:
    """
    Ranks candidate dishes based on user goals, target macros, and recent eating history.
    Attaches a match percentage and a custom Vietnamese rationale to each dish.
    """
    goal = user_profile.get("goal", "maintain")
    target_cal = user_profile.get("target_calories", 2000)
    target_p = user_profile.get("target_protein", 100)
    target_c = user_profile.get("target_carbs", 200)
    target_f = user_profile.get("target_fat", 65)

    # 1. Analyze flavor preferences and protein sources from recent meals
    recent_titles_str = " ".join([m.get("name", "").lower() for m in recent_meals])
    
    # Track counts of common protein keywords
    keywords = {
        "gà (chicken)": ["gà", "chicken"],
        "bò (beef)": ["bò", "beef"],
        "heo (pork)": ["heo", "pork", "thịt"],
        "tôm (shrimp)": ["tôm", "shrimp", "seafood", "cá", "fish"],
        "trứng (egg)": ["trứng", "egg"],
        "đậu/chay (tofu/veggie)": ["chay", "tofu", "đậu", "salad"]
    }
    
    recent_keyword_counts = {}
    for key, synonyms in keywords.items():
        count = sum(len(re.findall(rf"\b{syn}\b", recent_titles_str)) for syn in synonyms)
        if count > 0:
            recent_keyword_counts[key] = count

    ranked_results = []
    for dish in candidate_dishes:
        title = dish.get("title", "")
        desc = dish.get("description", "")
        dish_content = (title + " " + desc).lower()
        
        dish_cal = dish.get("calories", 0)
        dish_p = dish.get("protein", 0)
        dish_c = dish.get("carbs", 0)
        dish_f = dish.get("fat", 0)

        # A. Macro similarity score
        user_vector = [target_p, target_c, target_f]
        dish_vector = [dish_p, dish_c, dish_f]
        macro_sim = cosine_similarity(user_vector, dish_vector)

        # B. Variety factor (fatigue discount for recently consumed flavors)
        fatigue_penalty = 0.0
        matching_recent_key = None
        for key, synonyms in keywords.items():
            if any(syn in dish_content for syn in synonyms):
                # Penalty is proportional to how often they ate it recently (max 35% reduction)
                times_eaten = recent_keyword_counts.get(key, 0)
                if times_eaten > 0:
                    fatigue_penalty += min(times_eaten * 0.15, 0.35)
                    matching_recent_key = key

        # C. Goal-based density matching
        goal_score = 1.0
        if goal == "lose":
            # Lose weight: penalize high calorie-density dishes
            cal_density = dish_cal / max(dish_p + dish_c + dish_f, 1)
            if cal_density > 6.0:  # arbitrary threshold
                goal_score -= 0.2
            if dish_cal > (target_cal * 0.4):  # single meal exceeds 40% daily budget
                goal_score -= 0.15
        elif goal == "gain":
            # Gain weight: favor high protein and calorie density
            if dish_p > (target_p * 0.25):
                goal_score += 0.15
            if dish_cal > (target_cal * 0.2):
                goal_score += 0.1

        # D. Combine into final match score (0 - 100)
        base_score = (macro_sim * 60) + (goal_score * 40)
        final_score = base_score * (1.0 - fatigue_penalty)
        final_score = max(min(final_score, 100.0), 0.0)
        
        # Round score
        match_percentage = round(final_score)

        # E. Generate personalized Vietnamese rationale
        rationales = []
        if macro_sim > 0.85:
            rationales.append("Tỉ lệ dinh dưỡng (Macros) khớp hoàn hảo với mục tiêu của bạn.")
        elif dish_p > 25:
            rationales.append("Cung cấp lượng Protein dồi dào, rất thích hợp cho cơ bắp.")

        if matching_recent_key and fatigue_penalty > 0:
            rationales.append(f"Giúp thay đổi khẩu vị mới mẻ so với món {matching_recent_key.split(' ')[0]} bạn ăn gần đây.")
        else:
            rationales.append("Mang lại trải nghiệm ẩm thực đa dạng, đầy đủ vi chất.")

        if goal == "lose" and dish_cal < 450:
            rationales.append("Lượng calo thấp, hỗ trợ giảm cân hiệu quả mà không lo đói.")
        elif goal == "gain" and dish_cal > 500:
            rationales.append("Mật độ dinh dưỡng cao giúp bạn dễ dàng đạt mục tiêu tăng cân.")

        personalized_rationale = " ".join(rationales[:2])

        ranked_results.append({
            **dish,
            "match_score": match_percentage,
            "rationale": personalized_rationale
        })

    # Sort candidates by match score descending
    ranked_results.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked_results
