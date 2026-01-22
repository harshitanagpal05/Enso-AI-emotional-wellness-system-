import random

RECOMMENDATIONS_POOL = {
    "positive state": {
        "music": [
            {"content": "Zingaat (Bollywood)", "reason": "Unstoppable high energy and pure joy"},
            {"content": "Jai Ho (Bollywood)", "reason": "Ultimate victory anthem"},
            {"content": "Cruel Summer - Taylor Swift (Hollywood)", "reason": "High-energy pop anthem for a great mood"},
            {"content": "Dynamite - BTS (Hollywood Pop)", "reason": "Pure joy and catchy pop rhythm"},
            {"content": "London Thumakda (Bollywood)", "reason": "Celebratory wedding energy"},
            {"content": "Happy - Pharrell Williams (Hollywood)", "reason": "The universal anthem for feeling good"},
            {"content": "Khaabon Ke Parindey (Bollywood)", "reason": "Feel-good travel vibes"},
            {"content": "Uptown Funk - Bruno Mars (Hollywood)", "reason": "Funk and soul to keep the party going"},
            {"content": "Gallan Goodiyaan (Bollywood)", "reason": "Family celebration and togetherness"},
            {"content": "Levitating - Dua Lipa (Hollywood)", "reason": "Cosmic pop energy"},
            {"content": "Mauja Hi Mauja (Bollywood)", "reason": "Geet's infectious energy in song form"}
        ],
        "movie": [
            {"content": "Yeh Jawaani Hai Deewani (Bollywood)", "reason": "The vibe of travel, friendship and finding yourself"},
            {"content": "Friends (Hollywood Series)", "reason": "Ultimate comfort, laughter, and joy"},
            {"content": "Om Shanti Om (Bollywood)", "reason": "Pure Bollywood entertainment, color, and joy"},
            {"content": "Everything Everywhere All At Once (Hollywood)", "reason": "High energy, creativity, and emotional payoff"},
            {"content": "Dilwale Dulhania Le Jayenge (Bollywood)", "reason": "The ultimate classic Bollywood romance"},
            {"content": "The Grand Budapest Hotel (Hollywood)", "reason": "Whimsical, colorful, and visually delightful"},
            {"content": "3 Idiots (Bollywood)", "reason": "Friendship, laughter, and following your heart"},
            {"content": "Despicable Me (Hollywood)", "reason": "Pure animated joy and Minions!"},
            {"content": "Zindagi Na Milegi Dobara (Bollywood)", "reason": "Celebrate life, travel, and friendship"}
        ],
        "activity": [
            {"content": "Host a mini dance party", "reason": "Physical expression of joy boosts endorphins"},
            {"content": "Start a gratitude journal", "reason": "Focusing on what you're thankful for sustains happiness"},
            {"content": "Photography Walk", "reason": "Capturing beauty in your surroundings through your lens"},
            {"content": "Bake a new recipe", "reason": "Creating something sweet and rewarding"},
            {"content": "Compliment 3 strangers", "reason": "Spreading positivity makes you feel even better"},
            {"content": "Plan a dream vacation", "reason": "Visualization and planning fuel excitement"}
        ],
        "interactive": [
            {"content": "Solve a Jigsaw Puzzle", "reason": "Engaging and rewarding mental stimulation"},
            {"content": "Play a quick strategy game", "reason": "Sharpen your mind while in high spirits"},
            {"content": "Record a joyful voice note", "reason": "Capture this high energy for your future self"},
            {"content": "Do a 5-minute 'Laughter Yoga'", "reason": "Scientific way to multiply your current joy"}
        ],
        "sports": [
            {"content": "Football", "reason": "High-energy team sport to match your great mood"},
            {"content": "Basketball", "reason": "Dynamic and social way to stay active"},
            {"content": "Tennis", "reason": "Great for focus, agility, and a bit of healthy competition"},
            {"content": "Cricket", "reason": "Enjoyable team game to celebrate with friends"}
        ],
        "quote": [
            {"content": "Main apni favorite hoon.", "reason": "Geet's iconic self-love mantra from Jab We Met"},
            {"content": "Main udna chahta hoon... bas rukna nahi chahta.", "reason": "Bunny's relatable wanderlust from YJHD"},
            {"content": "All izz well.", "reason": "The ultimate positive mantra from 3 Idiots"},
            {"content": "Zindagi badi honi chahiye, lambi nahi.", "reason": "Deep life philosophy from Anand - quality over quantity"},
            {"content": "Koshish karne walon ki kabhi haar nahi hoti.", "reason": "Motivational wisdom - those who try never truly fail"},
            {"content": "Agar kisi cheez ko dil se chaaho, to puri kayanat usey tumse milane ki koshish mein lag jaati hai.", "reason": "The Secret's Hindi wisdom - the universe conspires for your dreams"},
            {"content": "Happiness is not something ready-made. It comes from your own actions.", "reason": "Dalai Lama's wisdom"},
            {"content": "Life is what happens when you're busy making other plans.", "reason": "John Lennon's reflection on living in the moment"},
            {"content": "The best is yet to come.", "reason": "An optimistic outlook for your bright future"},
            {"content": "Aaj se teri khushi meri khushi, teri mushkil meri mushkil.", "reason": "Beautiful promise of togetherness and support"},
            {"content": "Har ek friend zaroori hota hai, chahe wo life mein ho ya Facebook mein.", "reason": "3 Idiots' take on friendship and connections"}
        ]
    },
    "low": {
        "music": [
            {"content": "Kabira (Bollywood)", "reason": "Soothing, reflective, and deeply calming"},
            {"content": "Fix You - Coldplay (Hollywood)", "reason": "Soulful, comforting, and incredibly healing"},
            {"content": "Agar Tum Saath Ho (Bollywood)", "reason": "Cathartic and emotionally resonant"},
            {"content": "Someone Like You - Adele (Hollywood)", "reason": "Powerful emotional release through song"},
            {"content": "Kun Faya Kun (Bollywood)", "reason": "Spiritual, peaceful, and soul-cleansing"},
            {"content": "Stay - Rihanna (Hollywood)", "reason": "Raw and relatable emotional depth"},
            {"content": "Tujhe Kitna Chahne Lage (Bollywood)", "reason": "Soul-stirring and beautifully melodic"},
            {"content": "Ocean Eyes - Billie Eilish (Hollywood)", "reason": "Ethereal, calming, and atmospheric"}
        ],
        "movie": [
            {"content": "Jab We Met (Bollywood)", "reason": "The ultimate feel-good movie to lift your spirits"},
            {"content": "The Pursuit of Happyness (Hollywood)", "reason": "Deeply touching and incredibly motivational"},
            {"content": "Taare Zameen Par (Bollywood)", "reason": "Heartwarming story about uniqueness and care"},
            {"content": "Inside Out (Hollywood)", "reason": "A beautiful exploration of why every emotion matters"},
            {"content": "Queen (Bollywood)", "reason": "Empowering story of self-discovery and strength"},
            {"content": "Little Miss Sunshine (Hollywood)", "reason": "Quirky, dysfunctional, and ultimately hopeful"},
            {"content": "Dear Zindagi (Bollywood)", "reason": "A gentle hug for your soul and mind"},
            {"content": "Forrest Gump (Hollywood)", "reason": "Inspirational classic about the beauty of life's journey"}
        ],
        "activity": [
            {"content": "Gentle Nature Walk", "reason": "Fresh air and natural sights help clear the mind"},
            {"content": "Mood Boarding", "reason": "Visually expressing how you want to feel next"},
            {"content": "Watch a sunset", "reason": "A peaceful reminder that beauty exists in every cycle"},
            {"content": "Cozy Reading Session", "reason": "Lose yourself in a comforting, familiar book"},
            {"content": "Warm Bath with salts", "reason": "Physical comfort helps soothe emotional weight"},
            {"content": "Write a letter to yourself", "reason": "Self-compassion through written words"}
        ],
        "interactive": [
            {"content": "Deep Breathing Bubble Pop", "reason": "Focus on slow breathing to pop digital bubbles"},
            {"content": "Guided 'Loving-Kindness' Meditation", "reason": "Directing positive energy toward yourself"},
            {"content": "Draw your feelings with colors", "reason": "Abstract expression when words are hard"},
            {"content": "Listen to a 5-minute guided relaxation", "reason": "Gentle voice to guide you back to calm"}
        ],
        "sports": [
            {"content": "Swimming", "reason": "Rhythmic movement in water is deeply meditative"},
            {"content": "Yoga", "reason": "Gentle stretching to release tension held in the body"},
            {"content": "Table Tennis", "reason": "Light activity to gently regain focus"},
            {"content": "Leisurely Cycling", "reason": "Enjoy the breeze at your own slow pace"}
        ],
        "quote": [
            {"content": "Sometimes the wrong train takes you to the right station.", "reason": "Hopeful wisdom from The Lunchbox"},
            {"content": "Picture abhi baaki hai mere dost.", "reason": "A reminder that this isn't the end of your story - from Om Shanti Om"},
            {"content": "Tum agar khulke ro nahi sakogi... toh khulkar has kaise sakogi", "reason": "It is okay to let your emotions flow - from Dear Zindagi"},
            {"content": "Zindagi mein kuch banna hai, kuch kar dikhana hai, toh hamesha khud par bharosa rakhna hoga.", "reason": "Believe in yourself - motivational wisdom"},
            {"content": "Darr ke aage jeet hai.", "reason": "Victory lies beyond fear - from Dangal"},
            {"content": "Jab tak life hai, tab tak hope hai. Hope hai, tab tak life hai.", "reason": "As long as there's life, there's hope - from 3 Idiots"},
            {"content": "This too shall pass.", "reason": "A timeless reminder of your inner resilience"},
            {"content": "The sun will rise again tomorrow.", "reason": "Nature's promise of a new beginning"},
            {"content": "Dard kam hota hai khushi dene se.", "reason": "Small acts of kindness can heal your own heart"},
            {"content": "Agar aap apne aap ko change nahi kar sakte, toh kisi aur ko kaise change karoge?", "reason": "Change yourself first - from Queen"},
            {"content": "Har mushkil ke baad aasman mein khushi ka rang hota hai.", "reason": "After every difficulty, there's a rainbow of happiness"},
            {"content": "Zindagi ek safar hai suhana, yahan kal kya ho kisne jaana.", "reason": "Life is a beautiful journey - embrace the unknown"}
        ]
    },
    "frustrated and high stressed": {
        "music": [
            {"content": "Aarambh Hai Prachand (Bollywood)", "reason": "Powerful and high-energy to channel intensity"},
            {"content": "In the End - Linkin Park (Hollywood)", "reason": "Cathartic rock to release pent-up frustration"},
            {"content": "Zinda - Bhaag Milkha Bhaag (Bollywood)", "reason": "Motivational and intensely driving"},
            {"content": "Thunder - Imagine Dragons (Hollywood)", "reason": "Strong beat to help you find your focus"},
            {"content": "Apna Time Aayega (Bollywood)", "reason": "Raw energy and determination for the underdog"},
            {"content": "Sadda Haq (Bollywood Rockstar)", "reason": "An angsty and powerful anthem for release"},
            {"content": "Believer - Imagine Dragons (Hollywood)", "reason": "Channel your pain into inner strength"},
            {"content": "Lose Yourself - Eminem (Hollywood)", "reason": "Intense focus and rhythmic drive"}
        ],
        "movie": [
            {"content": "Rocky (Hollywood)", "reason": "The ultimate story of determination and grit"},
            {"content": "Gully Boy (Bollywood)", "reason": "Channelling life's frustrations into powerful art"},
            {"content": "Fight Club (Hollywood)", "reason": "A raw and intense exploration of release"},
            {"content": "Mary Kom (Bollywood)", "reason": "Turning anger into a focused fighting spirit"},
            {"content": "John Wick (Hollywood)", "reason": "High-octane action for some vicarious release"},
            {"content": "Whiplash (Hollywood)", "reason": "Focus, drive, and intense dedication"},
            {"content": "Lakshya (Bollywood)", "reason": "Finding purpose and focus amidst confusion"}
        ],
        "activity": [
            {"content": "Intense Cardio Session", "reason": "Sweat out the stress and reset your chemistry"},
            {"content": "Scribble wildly on paper", "reason": "A kinetic and safe way to release frustration"},
            {"content": "High-intensity cleaning", "reason": "Transform nervous energy into a clean space"},
            {"content": "Cold water splash", "reason": "A quick shock to the system to reset your 'vagus' nerve"},
            {"content": "Shout into a pillow", "reason": "A classic, immediate physical release"},
            {"content": "Tear up old newspapers", "reason": "Satisfying physical action to vent stress"}
        ],
        "interactive": [
            {"content": "Play a fast-paced 'Clicker' game", "reason": "Channel rapid energy into a simple digital task"},
            {"content": "5-minute 'Box Breathing'", "reason": "Scientific way to force your nervous system to calm down"},
            {"content": "Solve a Sudoku (Hard)", "reason": "Force your brain to switch from emotion to logic"},
            {"content": "Write a 'venting' letter (then delete it)", "reason": "Get it all out without consequences"}
        ],
        "sports": [
            {"content": "Boxing", "reason": "The best way to physically release pent-up anger"},
            {"content": "Sprinting", "reason": "Short bursts of max effort to clear your head"},
            {"content": "Squash", "reason": "Hitting a ball against a wall is incredibly cathartic"},
            {"content": "Weightlifting", "reason": "Channel your intensity into building physical power"}
        ],
        "quote": [
            {"content": "Control... Uday... Control.", "reason": "A bit of humor from 'Welcome' to lighten the mood"},
            {"content": "Don't angry me!", "reason": "Amitabh's classic line to acknowledge your mood"},
            {"content": "Agar aap apne gusse ko control nahi kar sakte, toh aap apni zindagi ko control nahi kar sakte.", "reason": "Control your anger to control your life"},
            {"content": "Gussa ek aag hai jo pehle aapko jalti hai, phir doosron ko.", "reason": "Anger is a fire that burns you first"},
            {"content": "Sadda Haq - Aithe Rakh!", "reason": "Rockstar's powerful anthem - claim your rights"},
            {"content": "Zinda hoon main, zinda rehna hai mujhe.", "reason": "I'm alive, and I need to stay alive - from Bhaag Milkha Bhaag"},
            {"content": "For every minute you are angry, you lose sixty seconds of happiness.", "reason": "Emerson's perspective"},
            {"content": "Anger is an acid that can do more harm to the vessel in which it is stored.", "reason": "Mark Twain's reflection"},
            {"content": "You are stronger than your current stress.", "reason": "A needed reminder of your capability"},
            {"content": "Apna time aayega - believe karo aur mehnat karo.", "reason": "Your time will come - believe and work hard"},
            {"content": "Gusse ko energy mein badal do, usey positive kaam mein lagao.", "reason": "Transform anger into energy for positive work"}
        ]
    },
    "anxious": {
        "music": [
            {"content": "Iktara (Bollywood)", "reason": "Soothing and peaceful melody to ground you"},
            {"content": "Weightless - Marconi Union (Ambient)", "reason": "Scientifically designed to reduce anxiety levels"},
            {"content": "Aashaayen (Bollywood)", "reason": "Hopeful and uplifting to counter worried thoughts"},
            {"content": "Safe and Sound - Taylor Swift (Hollywood)", "reason": "A comforting and protective atmospheric vibe"},
            {"content": "Tu Na Jaane Aas Paas Hai Khuda (Bollywood)", "reason": "Reassuring and spiritually grounding"},
            {"content": "Riverside - Agnes Obel (Hollywood)", "reason": "A peaceful, flowing melody for a busy mind"}
        ],
        "movie": [
            {"content": "Wake Up Sid (Bollywood)", "reason": "A relatable and calming vibe to help you ground"},
            {"content": "The Secret Life of Walter Mitty (Hollywood)", "reason": "Gently breaking out of your shell and overcoming fear"},
            {"content": "Lagaan (Bollywood)", "reason": "Overcoming big fears through teamwork and courage"},
            {"content": "My Neighbor Totoro (Hollywood/Studio Ghibli)", "reason": "Whimsical, peaceful, and ultimate comfort"},
            {"content": "Finding Nemo (Hollywood)", "reason": "A beautiful journey of overcoming constant anxiety"},
            {"content": "Piku (Bollywood)", "reason": "Grounded, realistic, and oddly comforting slice of life"}
        ],
        "activity": [
            {"content": "5-4-3-2-1 Grounding Exercise", "reason": "Brings your senses back to the safe 'now'"},
            {"content": "Weighted Blanket & Tea", "reason": "Physical sense of safety and internal warmth"},
            {"content": "Organize a small drawer", "reason": "Gaining small, manageable control over your space"},
            {"content": "Listen to rain sounds", "reason": "Predictable auditory input is very calming"},
            {"content": "Repeat a calming affirmation", "reason": "Safe repetitive thoughts to replace 'what ifs'"},
            {"content": "Mindful coloring", "reason": "Focus on staying within lines to slow down thoughts"}
        ],
        "interactive": [
            {"content": "Follow a digital 'Breathing Square'", "reason": "Visual guide to stabilize your breathing pattern"},
            {"content": "Play a 'Sorting' puzzle", "reason": "Simple, repetitive tasks help quiet an anxious mind"},
            {"content": "Guided Body Scan", "reason": "Focus on each part of your body to release hidden tension"},
            {"content": "Digital Journaling Prompt", "reason": "Structure your thoughts to make them less overwhelming"}
        ],
        "sports": [
            {"content": "Yoga (Vinyasa Flow)", "reason": "Mindful movement to ground your nervous energy"},
            {"content": "Tai Chi", "reason": "Slow, deliberate movements to build focus and calm"},
            {"content": "Long Distance Walking", "reason": "Rhythmic and grounding for a racing mind"},
            {"content": "Golf", "reason": "A quiet, focused, and steady environment"}
        ],
        "quote": [
            {"content": "All izz well.", "reason": "Rancho's simple yet powerful way to calm the heart from 3 Idiots"},
            {"content": "Darr ke aage jeet hai.", "reason": "A reminder that bravery is just on the other side - from Dangal"},
            {"content": "Ghabraana nahi hai, bas confidence se kaam karna hai.", "reason": "Don't panic, just work with confidence"},
            {"content": "Anxiety sirf ek feeling hai, yeh aap nahi hain.", "reason": "Anxiety is just a feeling, it's not who you are"},
            {"content": "Zindagi mein har problem ka solution hota hai, bas dhoondhna padta hai.", "reason": "Every problem has a solution, you just need to find it"},
            {"content": "Courage is not the absence of fear, but the triumph over it.", "reason": "Nelson Mandela"},
            {"content": "Worrying is like a rocking chair; it gives you something to do but gets you nowhere.", "reason": "Van Wilder's wisdom"},
            {"content": "Everything will be okay in the end. If it's not okay, it's not the end.", "reason": "John Lennon"},
            {"content": "Aasman se gira, khajur mein atka - kuch bhi ho sakta hai!", "reason": "Fell from sky, stuck in date palm - anything can happen! Stay positive"},
            {"content": "Jo dar gaya, samjho mar gaya.", "reason": "Whoever fears, consider them dead - face your fears"}
        ]
    },
    "strong dislike": {
        "music": [
            {"content": "Tum Se Hi (Bollywood)", "reason": "Gentle and pure melody to cleanse the mood"},
            {"content": "Pure Shores - All Saints (Hollywood)", "reason": "Ethereal and clean soundscape to reset your mind"},
            {"content": "Rainy Day (Lo-fi)", "reason": "Neutral and refreshing background noise"},
            {"content": "Saathiya Title Track (Bollywood)", "reason": "Lush and beautiful production to lift spirits"},
            {"content": "Strawberry Fields Forever (Hollywood)", "reason": "Dreamy and escapist classic"}
        ],
        "movie": [
            {"content": "Chef (Hollywood)", "reason": "A low-stakes, feel-good movie about passion and craft"},
            {"content": "Amélie (Hollywood)", "reason": "Beautifully shot and quirky, a visual palette cleanser"},
            {"content": "Dil Chahta Hai (Bollywood)", "reason": "Fresh, stylish, and a celebration of life"},
            {"content": "Ratatouille (Hollywood)", "reason": "A celebration of taste and finding beauty anywhere"},
            {"content": "The Lunchbox (Bollywood)", "reason": "Simple, elegant, and deeply human storytelling"}
        ],
        "activity": [
            {"content": "Declutter one small corner", "reason": "External order helps internal clarity"},
            {"content": "Scented candle or essential oils", "reason": "Refreshing the senses with pleasant aromas"},
            {"content": "Wash your face with cold water", "reason": "A quick physical and mental refresh"},
            {"content": "Listen to white noise", "reason": "Auditory palette cleanser to block out noise"},
            {"content": "Eat a fresh, crisp apple", "reason": "Sensory palette cleanser with a satisfying crunch"}
        ],
        "interactive": [
            {"content": "Digital 'Zen' Garden", "reason": "Raking sand for a moment of focus and peace"},
            {"content": "Play a color-sorting game", "reason": "Visually satisfying and mentally calming"},
            {"content": "List 5 things you can smell/touch", "reason": "Ground yourself in the immediate reality"},
            {"content": "Try a 2-minute mindful tasting", "reason": "Focus entirely on a single flavor"}
        ],
        "sports": [
            {"content": "Badminton", "reason": "Fast-paced and refreshing change of rhythm"},
            {"content": "Cycling", "reason": "Focus on the path ahead and the feeling of wind"},
            {"content": "Hiking", "reason": "Nature's way of refreshing your entire perspective"},
            {"content": "Bowling", "reason": "Casual and social way to reset your focus"}
        ],
        "quote": [
            {"content": "Don't let outside things ruin your vibe.", "reason": "A modern reminder of your inner power"},
            {"content": "Main apni favorite hoon.", "reason": "Always prioritize your own peace and self-worth - from Jab We Met"},
            {"content": "All izz well.", "reason": "A simple mantra to reset your internal state - from 3 Idiots"},
            {"content": "Zindagi mein kuch bhi ho sakta hai, bas positive raho.", "reason": "Anything can happen in life, just stay positive"},
            {"content": "The world is full of magical things, patiently waiting for our senses to grow sharper.", "reason": "W.B. Yeats"},
            {"content": "Shift your focus back to peace.", "reason": "Actionable advice for a cluttered mind"},
            {"content": "Kuch bhi ho, khud se pyaar karo - that's the most important thing.", "reason": "No matter what, love yourself - that's most important"}
        ]
    },
    "shocking and unexpected wonders": {
        "music": [
            {"content": "Mast Magan (Bollywood)", "reason": "Sweet and rhythmic for a pleasant surprise"},
            {"content": "A Thousand Years (Hollywood)", "reason": "Timeless, evocative, and beautifully surprising"},
            {"content": "Kashmir Main Tu Kanyakumari (Bollywood)", "reason": "Playful and unexpected high energy"},
            {"content": "Electric Love - BØRNS (Hollywood)", "reason": "Sparky and surprising pop sound"},
            {"content": "Skyfall - Adele (Hollywood)", "reason": "Grand, impactful, and full of wonder"}
        ],
        "movie": [
            {"content": "Inception (Hollywood)", "reason": "Mind-bending and full of brilliant surprises"},
            {"content": "The Prestige (Hollywood)", "reason": "The ultimate movie about the 'big reveal'"},
            {"content": "Andhadhun (Bollywood)", "reason": "A masterclass in unexpected twists and turns"},
            {"content": "Interstellar (Hollywood)", "reason": "A visual spectacle of cosmic wonder"},
            {"content": "Parasite (Hollywood/Global)", "reason": "A rollercoaster of genre-defying brilliance"}
        ],
        "activity": [
            {"content": "Try a random new hobby tutorial", "reason": "Embrace the spirit of novelty and learning"},
            {"content": "Take a different route home", "reason": "Discover small surprises in your daily life"},
            {"content": "Learn a 'Did you know?' fact", "reason": "Engage with the world's hidden wonders"},
            {"content": "Blind taste test of fruits", "reason": "Engage your senses in a surprising way"},
            {"content": "Write down a surprising dream", "reason": "Capture the wonders of your subconscious mind"}
        ],
        "interactive": [
            {"content": "Solve a riddle or brain teaser", "reason": "Mental agility to match your surprised state"},
            {"content": "Play a 'Hidden Object' game", "reason": "Focus on finding wonders in plain sight"},
            {"content": "Try a digital optical illusion", "reason": "Challenge your perception of what's real"},
            {"content": "Record your reaction", "reason": "Save this moment of wonder for later"}
        ],
        "sports": [
            {"content": "Bouldering", "reason": "Challenging and unexpected ways to test strength"},
            {"content": "Surfing", "reason": "Learn to ride the waves of life's surprises"},
            {"content": "Skateboarding", "reason": "Novelty, balance, and flow in motion"},
            {"content": "Parkour", "reason": "See the world as a playground of wonders"}
        ],
        "quote": [
            {"content": "Life is like a box of chocolates. You never know what you're gonna get.", "reason": "Forrest Gump"},
            {"content": "Kuch bhi ho sakta hai - life is full of surprises!", "reason": "Anything can happen - embrace the unexpected"},
            {"content": "Main apni favorite hoon.", "reason": "Be your own favorite surprise every day - from Jab We Met"},
            {"content": "All izz well.", "reason": "Stay grounded even in the most surprising moments - from 3 Idiots"},
            {"content": "Expect the unexpected.", "reason": "The best way to handle life's brilliant twists"},
            {"content": "The most beautiful thing we can experience is the mysterious.", "reason": "Albert Einstein"},
            {"content": "Zindagi ek surprise hai, bas enjoy karo!", "reason": "Life is a surprise, just enjoy it!"}
        ]
    },
    "stable and calm state": {
        "music": [
            {"content": "Kesariya (Bollywood)", "reason": "Beautiful melody and perfectly balanced energy"},
            {"content": "Perfect - Ed Sheeran (Hollywood)", "reason": "A smooth, steady, and heartwarming pop ballad"},
            {"content": "Raabta (Bollywood)", "reason": "Melodic, steady, and very easy to listen to"},
            {"content": "Stay - Justin Bieber (Hollywood)", "reason": "Catchy and upbeat for a productive, balanced day"},
            {"content": "Kun Faya Kun (Bollywood Rockstar)", "reason": "Spiritual and soul-cleansing for a calm state"}
        ],
        "movie": [
            {"content": "Dil Chahta Hai (Bollywood)", "reason": "The ultimate coming-of-age classic for a steady mood"},
            {"content": "Chef (Hollywood)", "reason": "A low-stakes, feel-good movie about passion and craft"},
            {"content": "Vivaah (Bollywood)", "reason": "Wholesome family values and heartwarming vibes"},
            {"content": "The Godfather (Hollywood)", "reason": "A masterpiece for when you have the focus for great cinema"},
            {"content": "Piku (Bollywood)", "reason": "Slice-of-life realism that feels deeply relatable"},
            {"content": "Pulp Fiction (Hollywood)", "reason": "Stylized, engaging, and a masterclass in storytelling"}
        ],
        "activity": [
            {"content": "Read a new book chapter", "reason": "Maintains steady cognitive engagement"},
            {"content": "Plan your upcoming week", "reason": "Use your stable mood for effective organization"},
            {"content": "Learn 5 new words in a new language", "reason": "Gentle mental expansion"},
            {"content": "Practice Calligraphy", "reason": "Focused, beautiful, and steady hand-eye coordination"},
            {"content": "Cook a slightly complex meal", "reason": "A rewarding process that requires balanced focus"}
        ],
        "interactive": [
            {"content": "Crossword or Wordle", "reason": "Engaging but relaxed mental challenge"},
            {"content": "Listen to an educational podcast", "reason": "Leverage your calm state for learning"},
            {"content": "Digital 'Zen' Garden", "reason": "Raking digital sand for a moment of focus"},
            {"content": "Mindful Sketching", "reason": "Drawing objects around you with detail"}
        ],
        "sports": [
            {"content": "Chess", "reason": "A strategic and calm mental workout"},
            {"content": "Archery", "reason": "Focus, precision, and mindfulness in every shot"},
            {"content": "Swimming (Laps)", "reason": "Steady, meditative, and physically rewarding"},
            {"content": "Billiards / Pool", "reason": "Focus, geometry, and a relaxed pace"}
        ],
        "quote": [
            {"content": "Babu Moshai, zindagi badi honi chahiye, lambi nahi.", "reason": "Deep life philosophy from 'Anand' - quality over quantity"},
            {"content": "Insaan ka motion uska emotion ke saath juda hota hai.", "reason": "Piku's relatable wisdom - motion is connected to emotion"},
            {"content": "All izz well.", "reason": "The ultimate calm mantra from 3 Idiots"},
            {"content": "Zindagi mein balance chahiye - thoda kaam, thoda aram.", "reason": "Life needs balance - some work, some rest"},
            {"content": "Seize the day, boys. Make your lives extraordinary.", "reason": "Dead Poets Society"},
            {"content": "Calmness is the cradle of power.", "reason": "Josiah Gilbert Holland"},
            {"content": "Be like water, my friend.", "reason": "Bruce Lee's philosophy of adaptable calm"},
            {"content": "Shanti se raho, sab theek ho jayega.", "reason": "Stay peaceful, everything will be okay"}
        ]
    }
}

# Track recently shown recommendations to ensure uniqueness
_recent_recommendations_cache = {}
MAX_CACHE_SIZE = 50

def get_recommendations(emotion, user_history=None):
    emotion_key = emotion.lower()
    
    # Fallback if emotion not in pool
    base_options = RECOMMENDATIONS_POOL.get(emotion_key, RECOMMENDATIONS_POOL["stable and calm state"])

    recommendations = []
    
    # Categories to include for a balanced mix
    categories = ["music", "movie", "activity", "interactive", "sports", "quote"]
    
    # Get recently shown items for this emotion to avoid repetition
    recent_items = _recent_recommendations_cache.get(emotion_key, [])
    
    for category in categories:
        if category in base_options:
            options = base_options[category]
            
            # Filter out recently shown items
            available_options = [opt for opt in options if opt["content"] not in recent_items]
            
            # If we've shown too many, reset and use all options
            if len(available_options) < 3:
                available_options = options
                recent_items = []  # Reset cache for this emotion
            
            # Select a varied number of items per category
            num_to_select = 2
            if category in ["quote", "interactive"]:
                num_to_select = 1
            elif category == "activity":
                num_to_select = 1
            
            # Ensure we don't select more than available
            num_to_select = min(len(available_options), num_to_select)
            
            if num_to_select > 0:
                selected = random.sample(available_options, num_to_select)
                for item in selected:
                    rec = {
                        "type": category,
                        "content": item["content"],
                        "reason": item["reason"]
                    }
                    recommendations.append(rec)
                    # Track this item as recently shown
                    recent_items.append(item["content"])
    
    # Update cache (keep only recent items)
    if len(recent_items) > MAX_CACHE_SIZE:
        recent_items = recent_items[-MAX_CACHE_SIZE:]
    _recent_recommendations_cache[emotion_key] = recent_items
    
    # Shuffle the final list to keep it fresh and unique
    random.shuffle(recommendations)
    
    return recommendations
