# -*- coding: utf-8 -*-
import csv, os
from pathlib import Path

PROJECT_ROOT = Path("/Users/commerciax-fs-1/Documents/proprietary_model_stt/our-stt-model")

# ============================================================
# STEP 1: Define all CORRECT script text for each block
# ============================================================

# --- ENGLISH v1 blocks (for amaan1.wav) ---
EN_V1_BLOCK1 = "Water. Table. Morning. Market. Doctor. Bicycle. Window. Yellow. Music. Tomorrow."
EN_V1_BLOCK2_PAIRS = "Ship, sheep. The sheep is near the ship. Live, leave. I live in the house and leave nearby. Bed, bad. The bad bed is in the room. Cut, cat, coat. The cat sat on the coat. Pin, pen. Put the pin and the pen in the bag. Thin, tin. The thin man has a tin box. Very, berry. The very sweet berry made him happy. Write, right, light. Turn on the light and write it right. Full, fool, pull. Don't be a fool, pull the door. Chip, ship, sip. He took a sip on the ship."

# --- ENGLISH v2 blocks ---
EN_V2 = {
    1: "Computer. Umbrella. Vegetable. Question. Elevator. Chocolate. Newspaper. Hospital. Calendar. Battery.",
    2: "The price is four point five dollars. Add one and a half cups of flour. She finished in second place. Today is the twenty first of the month. This building was made in nineteen ninety eight. The shop opens in twenty twenty seven. Three quarters of the class passed. The temperature dropped to minus two degrees. He ran ten point two kilometres. It is the third floor, room number three twelve. About sixty five percent of people agreed. The bag weighs two point eight kilos.",
    3: "A T M. P D F. U S B. C E O. F A Q. I D card. G P S. W H O. S M S. V I P.",
    4: "My address is twelve Green Lane, apartment four B. Send it to maria dot lopez at example dot com. The email is info at northwind dot org. Go to www dot bluebird dot net slash help. My username is kevin underscore ray ninety nine. The zip code is nine zero two one zero. Please write to forty five Oak Road, Leeds. The link ends with dot com slash contact.",
    5: "The meeting starts in five minutes. Please send me the report by noon. I will share my screen now. Can we move the call to Thursday? She is on leave until next Monday. The team finished the project early. I need to print two copies of this page. Let's check the schedule for next week. He is working from home today. Please sign the form at the bottom. The deadline is the end of the month. I'll get back to you after lunch.",
    6: "Open your books to page forty. Who knows the answer to this question? The exam will be on Friday morning. She reads a story to the class every day. Please write your name at the top. We have a science test next week. The teacher gave us some homework. He learned to count up to a hundred. Practice a little every day and you will improve.",
    7: "I would like to check my account balance. My card was declined at the shop. Please transfer the money today. I want to report a lost card. How long will the refund take? Can I speak to a manager? I have a problem with my order. My package has not arrived yet. I would like to change my delivery address. Thank you for calling, how can I help you? Please hold for a moment. Your request has been received.",
    8: "It looks like it will rain this evening. The sky is clear and the sun is bright. A cold wind is blowing from the north. There was heavy fog on the road this morning. The river is very high after the storm. Leaves turn red and yellow in autumn. It will be warm and sunny all weekend. The garden is full of flowers in spring. A big tree fell during the night. The mountains were covered with snow.",
    9: "Our team won the match by two goals. He plays football every Sunday. I like to paint and draw in my free time. She goes swimming three times a week. Let's play cards after dinner. The race starts at eight o'clock. I am learning to play the guitar. They went hiking in the hills. My hobby is taking photos of birds. He scored the last point of the game.",
    10: "Open the camera. Take a photo. Turn on the flashlight. Turn off Wi-Fi. Switch to dark mode. Increase the font size. Scan this document. Share this picture with Sarah. Delete the last message. Search for coffee shops near me. Show me my downloads. Update all my apps.",
    11: "Who is coming to the party? What time does the store close? When will the parcel arrive? Where did you put the charger? Why is the page not loading? How do I reset my password? Which bus goes to the airport? How many people are in the room? Whose bag is this? What is the best way to get there? Is there any milk left? Are you free this evening?",
    12: "Tom woke up early on Saturday. He made a cup of coffee and sat by the window. Outside, the street was quiet and a light rain was falling. The library is open from nine in the morning to six in the evening. You can borrow up to five books at a time. Please return them within two weeks. To make tea, first boil some water. Put a tea bag in a cup and pour the hot water over it. Wait for three minutes, then add milk if you like. The train to the city takes about forty minutes. It stops at four stations on the way. Tickets can be bought at the counter or on your phone.",
    13: "I think we should leave a bit earlier. Well, it's not that far from here. I mean, it was okay, but not great. Wait, no, I meant Tuesday, not Wednesday. So, I went there and nobody was home. It's about, let me think, maybe ten minutes away. Sorry, can you repeat the last part? Actually, let's make it seven, not six. I'm not sure, I'll have to check. Yeah, no, that's fine, go ahead.",
    14: "Explain how to make your favourite snack or drink. Describe the last time you went shopping. Talk about a person who has helped you. What is a normal day at work or school like for you? Describe the weather today and how it makes you feel. Talk about a place you would like to visit. Explain a game or sport to someone who has never played it. What is one thing technology does well and one thing it does badly?",
}

# --- ENGLISH v1 additional blocks ---
EN_V1 = {
    3: "My name is Aaron Mitchell. This is my sister Priya Sharma. Please call David or Emily. We are flying to London next week. He lives in Pune, near the river. The train goes from Delhi to Mumbai. I visited Paris, Rome and Tokyo last year. Let's meet at the Central Library. Our office is on Park Street in Kolkata. The nearest airport is in Chennai. I want a cup of tea with no sugar. Do you have this shirt in a bigger size? Where is the bathroom? Can I get the bill, please? The rice is a little too spicy. We need more soap and toothpaste. Please open the window, it is hot. My keys are not on the kitchen table. Can you put this in a paper bag? The fridge is almost empty. I forgot my umbrella at home. How much is one kilo of apples?",
    4: "Which platform does the train leave from? Turn left at the traffic light. Go straight for two hundred metres. Is this the right way to the station? I would like a window seat, please. How long does the journey take? Stop here, please. The road is closed because of the rain. Where can I buy a bus ticket? My flight was delayed by an hour. Turn on the light. Turn off the fan. Play some music. Pause the video. Set an alarm for six thirty. Call Mom. Send a message to John. Read my new messages. What is the time? Make the screen brighter. Show me the way home. Add bread to my shopping list.",
    5: "Hi, how are you? I am fine, thanks. And you? What did you do today? I can't come today, I am sorry. That sounds like a good idea. I don't know the answer. Could you say that one more time? Let's have lunch together. I will call you later tonight. Thank you so much for your help. Last weekend we went to the market, bought some fruit and vegetables, and then had lunch at a small cafe near the lake. If it rains tomorrow, we will stay at home and watch a movie instead of going to the park. My brother works at the bank, and he usually gets home around seven in the evening. I have been learning to cook for a few months, and now I can make simple dishes like pasta and fried rice. The children were playing in the garden while their parents were talking on the porch. When you reach the corner, you will see a blue building next to a small bakery.",
    6: "She sells seashells by the seashore. Red lorry, yellow lorry. Peter Piper picked a peck of pickled peppers. How much wood would a woodchuck chuck? Fresh French fried fish. My name is Sam — S, A, M. Thomas — T, H, O, M, A, S. Boston — B, O, S, T, O, N. My email starts with j, o, h, n dot k. The website is www dot example dot com. You can! You can? You can. This is great! I really don't like this. Oh no, I am late again! Describe your morning routine. Talk about your favourite food and how it is made. Tell a short story about a trip you remember. Describe your house or your neighbourhood. What do you like to do on weekends? Talk about a movie, song or book you like. Explain how you get to work or school. What would you change about your city if you could?",
}

# --- HINDI v3 blocks ---
HI_V3 = {
    1: "किताब। कमरा। बारिश। सुबह। दोस्त। शिक्षक। स्टेशन। नदी। पहाड़। तस्वीर। खाना। दुकान।",
    2: "पक्षी सुबह पेड़ पर चहचहा रहे हैं। भैया ने भारी बैग उठाकर बाहर रखा। चिड़िया छत पर बैठकर दाना चुग रही है। घोड़ा तेज़ दौड़कर खेत के पास पहुँच गया। ठंड बढ़ने पर लोग गर्म कपड़े पहनने लगे। झरने के पास बैठकर सभी ने पानी पिया। ज्ञानी व्यक्ति ध्यान और धैर्य से काम करता है। शिक्षक ने प्रश्न का उत्तर ध्यान से समझाया। श्रमिक सुबह जल्दी काम पर पहुँच गए। प्रार्थना के बाद सभी श्रद्धालु मंदिर से बाहर निकले।",
    3: "फोन नंबर: ९७६४१ ८२५३०। फोन नंबर: ८८३२७ ४१६९५। ओटीपी: ६३९२४८। ओटीपी: ०५८३१७। पिन कोड: ४०००७६। पिन कोड: ५६००३४। वाहन नंबर: एमएच १२ सीडी ४५६७। बारह अंकों का काल्पनिक नंबर: ७४१२ ०५८६ ३१९४।",
    4: "इस महीने मेरा बिजली का बिल दो हजार तीन सौ बीस रुपये आया। उसने नई बाइक पचहत्तर हजार रुपये में खरीदी। आज तारीख अट्ठाईस सितंबर है। छुट्टियाँ दस नवंबर से शुरू होंगी। बस सुबह नौ बजकर बीस मिनट पर आएगी। शाम साढ़े छह बजे डॉक्टर से मिलने जाना है। रात पौने दस बजे मुझे वापस घर पहुँचना है। मुझे चार मीटर लंबी रस्सी चाहिए। आज अधिकतम तापमान बत्तीस डिग्री रहेगा। बाजार से दो किलो चावल और आधा किलो दाल लाना। इस खाते पर ब्याज की दर सात प्रतिशत है। यहाँ से नासिक लगभग एक सौ पचास किलोमीटर दूर है।",
    5: "हम अगले महीने अहमदाबाद जा रहे हैं। नागपुर से भोपाल की ट्रेन सुबह निकलती है। उदयपुर की झीलें बहुत सुंदर हैं। कोलकाता में विक्टोरिया मेमोरियल देखने गए थे। साबरमती नदी अहमदाबाद से होकर गुजरती है। मेरे सहकर्मी का नाम रोहित शर्मा है। नेहा और अमित आज शाम आएँगे। प्रिया, मोहित और करण एक ही कक्षा में पढ़ते हैं। शिमला और मनाली दोनों लोकप्रिय पर्यटन स्थल हैं। वडाला, घाटकोपर और कुर्ला मुंबई के इलाके हैं।",
    6: "मेरे phone का charger घर पर रह गया है। इस document को email में attach कर देना। Wi-Fi की speed आज बहुत slow है। कल की meeting सुबह दस बजे होगी। मैंने नया keyboard online order किया है। Google Maps में nearest restaurant खोजो। मेरे account में payment अभी processing में है। UPI से payment करने के लिए QR code scan करो। Login करने के बाद अपना mobile number verify करें। YouTube पर यह video बहुत तेजी से viral हो गया। इस website में एक small bug मिला है। Zoom meeting का link मुझे WhatsApp पर भेज दो।",
    7: "कमरे की खिड़की खोलो। रसोई की लाइट चालू करो। पंखे की speed तीन पर करो। अगला वीडियो चलाओ। आवाज़ थोड़ा बढ़ाओ। पापा को फोन मिलाओ। ऑफिस जाने का रास्ता दिखाओ। शाम सात बजे का reminder लगाओ। आज की मुख्य खबरें पढ़कर सुनाओ। Bluetooth चालू करो। कल सुबह बारिश होगी क्या? खरीदारी की सूची में साबुन और फल जोड़ो।",
    8: "आज दिन कैसा रहा? नाश्ते में क्या खाया? चलो, थोड़ी देर बाहर घूमने चलते हैं। मैं आज ऑफिस से थोड़ा देर से निकलूँगा। आज मौसम काफी सुहावना है, है ना? आप कहाँ जा रहे हैं और कितने बजे लौटेंगे? मुझे यह जगह पहली बार देखने को मिली। ज़रा रुकिए, मैं अभी आता हूँ। इस रविवार को घर पर मिलकर खाना बनाते हैं। आपकी मदद के लिए बहुत धन्यवाद।",
    9: "कल सुबह मैं जल्दी उठा, नाश्ता किया, कुछ जरूरी काम पूरे किए और फिर दोपहर में दोस्तों से मिलने बाहर चला गया। अगर मौसम साफ रहा तो हम सप्ताहांत में पास के पहाड़ों पर घूमने जाने की योजना बना रहे हैं। मैंने तय किया है कि रोज़ थोड़ा समय पढ़ने और नई चीजें सीखने के लिए जरूर निकालूँगा। शहर में वाहनों की संख्या इतनी बढ़ गई है कि व्यस्त समय में छोटी दूरी तय करने में भी काफी समय लग जाता है। माँ हमेशा समझाती थीं कि किसी भी काम को जल्दी खत्म करने से ज्यादा जरूरी है उसे सही तरीके से पूरा करना। त्योहार के दिन सुबह से ही घर में सफाई, सजावट, खाना बनाने और मेहमानों के आने की तैयारी शुरू हो जाती है। अगर तुम समय पर स्टेशन पहुँच जाओ तो टिकट लेकर प्लेटफॉर्म पर मेरा इंतजार करना। नई नौकरी शुरू करने से पहले मैंने कंपनी, टीम और काम की जिम्मेदारियों के बारे में अच्छी तरह जानकारी ली।",
    10: "चंदा चमका, चाँदनी छाई, चिड़िया चहकी। पका पपीता, कच्चा पपीता, पपीते का पक्का पेड़। टेढ़ी टोपी पहनकर टिंकू टहलने निकला। गीता ने गर्म गाजर का हलवा जल्दी-जल्दी खाया। राधा रोज़ रंग-बिरंगे रुमाल रखती है। झटपट झोला उठाकर झील की ओर चलो।",
    11: "मेरा नाम रिया है — र, इ, य, आ। मुंबई — म, ु, ं, ब, ई। किताब — क, ि, त, ा, ब। मेरे ईमेल का नाम r-i-y-a यानी आर, आई, वाई, ए। वेबसाइट: w-w-w डॉट s-a-m-p-l-e डॉट i-n। पासवर्ड में capital A, फिर number 7 और small x है।",
    12: "तुम यहाँ हो! तुम यहाँ हो? तुम यहाँ हो। अरे वाह, तुमने बहुत अच्छा किया! मुझे यह बात बिल्कुल अच्छी नहीं लगी। ओह नहीं, मेरा फोन फिर से बंद हो गया! सच में, यह खबर बहुत अच्छी है! क्या तुमने मेरा संदेश पढ़ा?",
    13: "अपने शहर में घूमने के लिए सबसे अच्छी जगह के बारे में बताइए। अपनी पसंदीदा चाय या नाश्ता कैसे बनता है, समझाइए। स्कूल या कॉलेज की कोई मजेदार याद सुनाइए। आपने हाल में कोई यात्रा की हो আগের हो तो उसके बारे में बताइए। अपने काम या पढ़ाई में अभी कौन-सा काम सबसे ज्यादा कर रहे हैं? किसी पसंदीदा फिल्म, गाने या किताब के बारे में बताइए। अगले महीने आप क्या करना चाहते हैं? मोबाइल और इंटरनेट आपके रोज़मर्रा के जीवन में कैसे काम आते हैं? अपने किसी दोस्त या परिवार के सदस्य के बारे में बताइए। ऐसा कोई कौशल बताइए जिसे आप भविष्य में सीखना चाहते हैं।",
}

# --- HINDI v2 blocks ---
HI_V2 = {
    1: "पानी। घर। चाय। रास्ता। खुशी। विद्यार्थी। गाड़ी। बाजार। आकाश। मोबाइल।",
    2: "किसान खेत में हल चला रहा है। घड़ी में ठीक-ठीक टक-टक की आवाज़ आई। बच्चे पेड़ पर चढ़कर फल तोड़ रहे हैं। भगवान को धूप और दीपक चढ़ाया। बच्चे ढोल और ढोलक की ताल पर नाचे। ठंडी हवा चलने से पत्ते गिर पड़े। मेरे पास ज्ञान, ध्यान और धैर्य तीनों हैं। कृष्ण और ऋषि की कहानी दादी ने सुनाई। श्रावण के महीने में श्रद्धालु शिव मंदिर जाते हैं। नल वाले पुराने मकान में काली बिल्ली आँगन में घूमती है।",
    3: "फोन नंबर: ९८२५० १२३४५। फोन नंबर: ७८७८६ ५४३२१। ओटीपी: ४८२९१७। ओटीपी: ०७३५६१। पिन कोड: ३८०००१। पिन कोड: ३९५००७। वाहन नंबर: डीएल ०५ एबी १२३४। आधार जैसा बारह अंकों का काल्पनिक नंबर: २३४५ ६७८९ ०१२३।",
    4: "मेरा बिल एक हजार आठ सौ पचास रुपये आया। उसने साढ़े बारह लाख रुपये में जमीन बेची। आज तारीख पंद्रह अगस्त है। परीक्षा दो मार्च से शुरू होगी। ट्रेन शाम सात बजकर पैंतीस मिनट पर चलेगी। सुबह सवा आठ बजे मीटिंग है। दोपहर पौने तीन बजे फोन करना। यह कपड़ा ढाई मीटर चाहिए। तापमान पैंतीस डिग्री तक पहुँच गया है। दुकान से तीन किलो आलू और डेढ़ किलो प्याज लिए। ब्याज की दर साढ़े आठ प्रतिशत है। दिल्ली से जयपुर लगभग दो सौ साठ किलोमीटर दूर है।",
    5: "हम इस साल जयपुर जा रहे हैं। मुंबई से पुणे तक की टिकट बुक करो। वाराणसी की साड़ियाँ बहुत प्रसिद्ध हैं। दिल्ली में लाल किला देखने गए थे। पोरबंदर महात्मा गांधी का जन्मस्थान है। मेरे दोस्त का नाम हितेश पटेल है। मीनाबेन और जयेशभाई कल आएँगे। भूमिका, कुणाल और राहुल तीनों साथ पढ़ते थे। नैनीताल और मसूरी दोनों घूमने लायक जगहें हैं। अंधेरी, बांद्रा और दादर मुंबई के इलाके हैं।",
    6: "मेरे phone की battery बहुत जल्दी खत्म हो जाती है। इस file को email के जरिए भेज दो। Wi-Fi का password क्या है? आज की meeting cancel हो गई है। मैंने नया laptop order किया है, कल delivery होगी। Google Maps में पास का petrol pump खोजो। मेरे account में payment अभी pending दिख रहा है। UPI से पैसे भेजने के लिए QR code scan करो। Online form भरते समय mobile number verify करना होगा। Instagram पर यह reel बहुत viral हो गई। इस code में एक bug है, मैं इसे fix कर रहा हूँ। Zoom पर link share करो, मैं अभी join करता हूँ।",
    7: "पंखा चालू करो। बेडरूम की लाइट बंद करो। एसी का तापमान छब्बीस डिग्री करो। अगला गाना बजाओ। आवाज़ कम करो। मम्मी को फोन लगाओ। घर जाने का रास्ता दिखाओ। सुबह छह बजे का अलार्म लगाओ। आज की खबरें सुनाओ। टॉर्च चालू करो। कल का मौसम कैसा रहेगा? खरीदारी की सूची में दूध और ब्रेड जोड़ो।",
    8: "कैसे हो? मज़े में? आज खाने में क्या बनाया है? चलो, चाय पीने चलते हैं। मैं थोड़ा देर से पहुँचूँगा, आप खाना खा लेना। आज तो बहुत गर्मी है, है ना? आपका नाम क्या है और आप कहाँ रहते हैं? मुझे यह चीज़ बहुत पसंद आई। भाई, ज़रा धीरे चलाओ। अगले रविवार को हम सब साथ में खाना खाने मिलें। आपका बहुत-बहुत धन्यवाद, आपने बहुत मदद की।",
    9: "कल शाम हम सभी दोस्त इकट्ठा हुए, थोड़ी बातें कीं, चाय-नाश्ता किया और फिर देर रात तक पुराने दिनों की यादें ताज़ा कीं। अगर बारिश समय पर हो जाए तो इस साल किसानों को अच्छी फसल मिलने की उम्मीद है, वरना दाम बढ़ सकते हैं। मैंने काफी समय से सोचा था कि नई भाषा सीखनी चाहिए, इसलिए अब रोज़ आधा घंटा पढ़ाई करने का फैसला किया है। शहर में ट्रैफिक इतना बढ़ गया है कि पंद्रह मिनट का रास्ता तय करने में कभी-कभी एक घंटा लग जाता है। दादी हमेशा कहती थीं कि मेहनत का फल देर से मिलता है, लेकिन मीठा मिलता है। त्योहारों के दिनों में घर को रंगोली और तोरण से सजाया जाता है और सभी परिवार वाले मिलकर मिठाई बनाते हैं।",
    10: "कच्चा पापड़, पक्का पापड़। छः छोटी छाबड़ियों में छः छिले हुए चने। चंदू के चाचा ने चाँदी की चम्मच से चटनी चखी। गधे ने गाजर गपागप खाकर गाना गाया। ढब्बू ढोलकिया ढोल बजाए, ढोल की थाप पर ढेला डोले।",
    11: "मेरा नाम सेरा है — स, ए, र, आ। दिल्ली — द, ि, ल, ्, ल, ी। भारत — भ, ा, र, त। मेरे ईमेल का नाम s-e-r-a यानी एस, ई, आर, ए। वेबसाइट: w-w-w डॉट e-x-a-m-p-l-e डॉट c-o-m।",
    12: "तुम आ गए! तुम आ गए? तुम आ गए। वाह, यह तो बहुत अच्छा है! मुझे यह बिल्कुल पसंद नहीं आया। अरे, फिर से देर हो गई!",
    13: "अपने शहर या गाँव की किसी खास जगह के बारे में बताइए। अपनी पसंदीदा डिश कैसे बनती है, यह समझाइए। बचपन की कोई यादगार घटना सुनाइए। आपने आखिरी बार कोई त्योहार कैसे मनाया था? अपने काम या पढ़ाई में आजकल क्या चल रहा है? किसी फिल्म, गाने या किताब के बारे में अपनी राय बताइए। अगले हफ्ते की आपकी क्या योजना है? टेक्नोलॉजी आपके जीवन में किस तरह मदद करती है या परेशान करती है?",
}

# --- GUJARATI v2 blocks ---
GU_V2 = {
    1: "પાણી। ઘર। ચા। રસ્તો। ખુશી। વિદ્યાર્થી। ગાડી। બજાર। આકાશ। મોબાઇલ।",
    2: "ખેડૂત ખેતરમાં હળ ચલાવે છે। ઘડિયાળમાં ઠીક ઠીક ટકોરા પડ્યા। છોકરાઓ ઝાડ પર ચઢીને ફળ તોડે છે। ભગવાનને ધૂપ અને દીવો ધરાવ્યો। બાળકો ઢોલ અને ઢોલકના તાલે નાચ્યાં। ઠંડો પવન ફૂંકાતાં પાંદડાં ખરી પડ્યાં। મારી પાસે જ્ઞાન, ધ્યાન અને ધીરજ ત્રણેય છે। કૃષ્ણ અને ઋષિની વાર્તા દાદીએ સંભળાવી। શ્રાવણ મહિનામાં શ્રદ્ધાળુઓ શિવમંદિરે જાય છે। નળિયાંવાળા મકાનમાં કાળી બિલાડી ફળિયામાં ફરે છે।",
    3: "ફોન નંબર: ૯૮૨૫૦ ૧૨૩૪૫। ફોન નંબર: ૭૮૭૮૬ ૫૪૩૨૧। OTP: ૪૮૨૯૧૭। OTP: ૦૭૩૫૬૧। પિન કોડ: ૩૮૦૦૦૧। પિન કોડ: ૩૯૫૦૦૭। વાહન નંબર: GJ ૦૫ AB ૧૨૩૪। આધાર જેવો બાર આંકડાનો કાલ્પનિક નંબર: ૨૩૪૫ ૬૭૮૯ ૦૧૨૩।",
    4: "મારું બિલ એક હજાર આઠસો પચાસ રૂપિયા આવ્યું। તેણે સાડા બાર લાખ રૂપિયામાં જમીન વેચી। આજે તારીખ પંદર ઑગસ્ટ છે। પરીક્ષા બે માર્ચથી શરૂ થશે। ટ્રેન સાંજે સાત ને પાંત્રીસે ઊપડશે। સવારે સવા આઠ વાગ્યે મીટિંગ છે। બપોરે પોણા ત્રણ વાગ્યે ફોન કરજો। આ કાપડ અઢી મીટર જોઈએ છે। તાપમાન પાંત્રીસ ડિગ્રી સુધી પહોંચ્યું છે। દુકાનમાં ત્રણ કિલો બટાકા અને દોઢ કિલો ડુંગળી લીધી। વ્યાજનો દર સાડા આઠ ટકા છે। અમદાવાદથી રાજકોટ લગભગ બસો સાઠ કિલોમીટર દૂર છે।",
    5: "અમે આ વર્ષે ભાવનગર જઈ રહ્યા છીએ। સુરતથી વડોદરા સુધીની ટિકિટ બુક કરો। જામનગરની બાંધણી બહુ પ્રખ્યાત છે। ગાંધીનગરમાં અક્ષરધામ મંદિર જોવા ગયા હતા। પોરબંદર મહાત્મા ગાંધીનું જન્મસ્થળ છે। મારા મિત્રનું નામ હિતેશ પટેલ છે। મીનાબેન અને જયેશભાઈ કાલે આવશે। ભૂમિકા, કૃણાલ અને રાહુલ ત્રણેય સાથે ભણતા હતા। સાપુતારા અને ગિરનાર બંને ફરવા જેવી જગ્યાઓ છે। નવરંગપુરા, સેટેલાઇટ અને મણિનગર અમદાવાદના વિસ્તારો છે।",
    6: "મારા phone ની battery બહુ જલદી ઊતરી જાય છે। આ file ને email દ્વારા મોકલી આપો। Wi-Fi નો password શું છે? આજની meeting cancel થઈ ગઈ છે। મેં નવો laptop order કર્યો છે, કાલે delivery થશે। Google Maps માં નજીકનું petrol pump શોધો। મારા account માં payment હજી pending દેખાય છે। UPI થી પૈસા મોકલવા માટે QR code scan કરો। Online form ભરતી વખતે mobile number verify કરવો પડશે। Instagram પર આ reel બહુ viral થઈ ગઈ। આ code માં એક bug છે, હું તેને fix કરું છું। Zoom પર link share કરો, હું હમણાં join કરું છું।",
    7: "પંખો ચાલુ કરો। બેડરૂમની લાઇટ બંધ કરો। એસીનું તાપમાન છવ્વીસ ડિગ્રી કરો। આગલું ગીત વગાડો। અવાજ ઓછો કરો। મમ્મીને ફોન લગાવો। ઘરે જવાનો રસ્તો બતાવો। સવારે છ વાગ્યાનું એલાર્મ મૂકો। આજના સમાચાર સંભળાવો। ટૉર્ચ ચાલુ કરો। કાલનું હવામાન કેવું રહેશે? ખરીદીની યાદીમાં દૂધ અને બ્રેડ ઉમેરો।",
    8: "કેમ છો? મજામાં? આજે જમવામાં શું બનાવ્યું છે? ચાલો, ચા પીવા જઈએ। હું થોડો મોડો પડીશ, તમે જમી લેજો। આજે તો બહુ ગરમી છે, નહીં? તમારું નામ શું છે અને તમે ક્યાં રહો છો? મને આ વસ્તુ બહુ ગમી। ભાઈ, જરા ધીમે ચલાવો। આવતા રવિવારે આપણે બધા સાથે જમવા મળીએ। તમારો બહુ બહુ આભાર, તમે ખૂબ મદદ કરી।",
    9: "ગઈકાલે સાંજે અમે બધા મિત્રો ભેગા થયા, થોડી વાતો કરી, ચા-નાસ્તો કર્યો અને પછી મોડી રાત સુધી જૂના દિવસોની યાદો તાજી કરી। જો વરસાદ સમયસર પડે તો આ વર્ષે ખેડૂતોને સારો પાક મળવાની આશા છે, નહીંતર ભાવ વધી શકે છે। મેં ઘણા સમયથી વિચાર્યું હતું કે નવી ભાષા શીખવી જોઈએ, એટલે હવે રોજ અડધો કલાક અભ્યાસ કરવાનું નક્કી કર્યું છે। શહેરમાં ટ્રાફિક એટલો વધી ગયો છે કે પંદર મિનિટનો રસ્તો કાપવામાં ક્યારેક કલાક નીકળી જાય છે। દાદીમા હંમેશાં કહેતા કે મહેનતનું ફળ મોડું મળે છે, પણ મીઠું મળે છે। તહેવારોના દિવસોમાં ઘરને રંગોળી અને તોરણથી શણગારવામાં આવે છે અને બધા પરિવારજનો સાથે મળીને મીઠાઈ બનાવે છે।",
    10: "કાચો પાપડ પાકો પાપડ। છ છાબડીમાં છ છાલવાળાં છત્રી। ચંદુના ચાચાએ ચાંદીની ચમચીથી ચટણી ચાખી। ગધેડાએ ગાજર ગળચાવીને ગીત ગાયું। ઢબુ ઢોલી ઢોલ ઢબૂકાવે, ઢેલ ઢમકતી ચાલે।",
    11: "મારું નામ સેરા છે — સ, એ, ર, આ। અમદાવાદ — અ, મ, દ, આ, વ, આ, દ। ગુજરાત — ગ, ઉ, જ, ર, આ, ત। મારા ઈમેઇલનું નામ s-e-r-a એટલે એસ, ઈ, આર, એ। વેબસાઇટ: w-w-w ડૉટ e-x-a-m-p-l-e ડૉટ c-o-m।",
    12: "તમે આવી ગયા! તમે આવી ગયા? તમે આવી ગયા। વાહ, આ તો બહુ સરસ છે! મને આ બિલકુલ ન ગમ્યું। અરે, ફરીથી મોડું થઈ ગયું!",
    13: "તમારા શહેર કે ગામની કોઈ ખાસ જગ્યા વિશે જણાવો। તમારી મનપસંદ વાનગી કેવી રીતે બને છે તે સમજાવો। બાળપણની કોઈ યાદગાર ઘટના કહો। છેલ્લે તમે કોઈ તહેવાર કેવી રીતે ઉજવ્યો હતો? તમારા કામ કે અભ્યાસમાં આજકાલ શું ચાલે છે? કોઈ ફિલ્મ, ગીત કે પુસ્તક વિશે તમારો અભિપ્રાય આપો। આવતા અઠવાડિયાની તમારી યોજના શું છે? ટેક્નોલૉજી તમારા જીવનમાં કેવી રીતે મદદ કરે છે કે હેરાન કરે છે?",
}

# --- GUJARATI APPS blocks ---
GU_APPS = {
    "A": "AnyDesk બંધ કરો। મારા માટે AnyDesk બંધ કરી દો। શું AnyDesk હજી ચાલુ છે? Brave ખોલો। Brave Browser બંધ કરો। Safari પર જાઓ। Safari બંધ કરો। WhatsApp ખોલો। WhatsApp બંધ કરો। Slack ખોલો। Visual Studio Code ખોલો। Xcode બંધ કરો। Cursor ખોલો। Google Chrome બંધ કરો। Microsoft Outlook ખોલો। Microsoft Word બંધ કરો। KDE Connect ખોલો। ChatGPT બંધ કરો। App Store ખોલો। Finder ની વિન્ડો બંધ કરો।",
    "B": "Brave અને Safari બંધ કરો। Brave અને Safari ને બંધ કરી દો। બંને બંધ કરો, Brave અને Safari। Slack અને WhatsApp બંધ કરો। AnyDesk અને Postman બંધ કરો। Safari, Brave અને Chrome બંધ કરો। Safari સિવાય બધું બંધ કરો। મારા બધા browsers બંધ કરો। અટકી જાઓ। રહેવા દો। ફરીથી કરો।",
    "D": "પંદર મિનિટનો ટાઈમર સેટ કરો। સાડા ચાર વાગ્યે મને યાદ અપાવજો। ત્રેવીસમી તારીખે મારા કેલેન્ડરમાં શું છે? કાલે સવારે પોણા સાત વાગ્યે મને જગાડજો। તેની કિંમત બે હજાર ચારસો રૂપિયા છે। લગભગ વીસ મિનિટમાં મને પાછો ફોન કરજો। મીટિંગ શુક્રવાર, ત્રીજી તારીખે ખસેડાઈ છે। એ ઓગણીસ દશાંશ પાંચ ટકા છે। મારી ફ્લાઇટ રાત્રે અગિયાર ને પચાસ વાગ્યે છે। બત્રીસ અને અડતાલીસ ઉમેરો।",
    "E": "આજે હવામાન કેવું છે? શું તમે મને તેનો સારાંશ આપી શકો? તમે હમણાં જે કહ્યું તે મને સમજાયું નહીં। ફરીથી કહો, થોડું ધીમેથી। મારો મતલબ browser હતો, editor નહીં।",
}

# ============================================================
# STEP 2: Build the CORRECT mapping for each audio file
# ============================================================

correct_labels = {
    # --- ENGLISH (v1 batch: amaan1, amaan3-6, amaan_2_7) ---
    "data/processed/amaanvoice/englishvoice/amaan1.wav": (EN_V1_BLOCK1 + " " + EN_V1_BLOCK2_PAIRS, "en"),
    "data/processed/amaanvoice/englishvoice/amaan3.wav": (EN_V1[3], "en"),
    "data/processed/amaanvoice/englishvoice/amaan4.wav": (EN_V1[4], "en"),
    "data/processed/amaanvoice/englishvoice/amaan5.wav": (EN_V1[5], "en"),
    "data/processed/amaanvoice/englishvoice/amaan6.wav": (EN_V1[6], "en"),
    "data/processed/amaanvoice/englishvoice/amaan_2_7.wav": (EN_V2[13] + " " + EN_V2[14], "en"),
    
    # --- ENGLISH (v2 batch: amaan2, amaan2_1-6, aman2_4) ---
    "data/processed/amaanvoice/englishvoice/amaan2.wav": (EN_V2[2], "en"),  # only numbers block recorded
    "data/processed/amaanvoice/englishvoice/amaan2_1.wav": (EN_V2[1] + " " + EN_V2[2], "en"),
    "data/processed/amaanvoice/englishvoice/amaan2_2.wav": (EN_V2[3] + " " + EN_V2[4], "en"),
    "data/processed/amaanvoice/englishvoice/amaan2_3.wav": (EN_V2[5] + " " + EN_V2[6], "en"),
    "data/processed/amaanvoice/englishvoice/amaan2_5.wav": (EN_V2[9] + " " + EN_V2[10], "en"),
    "data/processed/amaanvoice/englishvoice/amaan2_6.wav": (EN_V2[11] + " " + EN_V2[12], "en"),
    "data/processed/amaanvoice/englishvoice/aman2_4.wav": (EN_V2[7] + " " + EN_V2[8], "en"),
    
    # --- GUJARATI (apps batch: guj_1, guj_2) ---
    "data/processed/amaanvoice/gujarativoice/amaan_guj_1.wav": (GU_APPS["A"], "gu"),
    "data/processed/amaanvoice/gujarativoice/amaan_guj_2.wav": (GU_APPS["B"], "gu"),
    
    # --- GUJARATI (apps batch: guj_3 = apps D + E) ---
    "data/processed/amaanvoice/gujarativoice/amaan_guj_3.wav": (GU_APPS["D"] + " " + GU_APPS["E"], "gu"),
    
    # --- GUJARATI (v2 batch: guj_4 through guj_9) ---
    "data/processed/amaanvoice/gujarativoice/amaan_guj_4.wav": (GU_V2[1] + " " + GU_V2[2], "gu"),
    "data/processed/amaanvoice/gujarativoice/amaan_guj_5.wav": (GU_V2[3], "gu"),
    "data/processed/amaanvoice/gujarativoice/amaan_guj_6.wav": (GU_V2[4] + " " + GU_V2[5], "gu"),
    "data/processed/amaanvoice/gujarativoice/Amaan_guj_7.wav": (GU_V2[6] + " " + GU_V2[7], "gu"),
    "data/processed/amaanvoice/gujarativoice/amaan_guj_8.wav": (GU_V2[8] + " " + GU_V2[9], "gu"),
    "data/processed/amaanvoice/gujarativoice/amaan_guj_9.wav": (GU_V2[10] + " " + GU_V2[11] + " " + GU_V2[12], "gu"),
    
    # --- HINDI v3 batch (hindi_1 through hin_8) ---
    "data/processed/amaanvoice/hindivoice/amaan_hindi_1.wav": (HI_V3[1] + " " + HI_V3[2], "hi"),
    "data/processed/amaanvoice/hindivoice/aman_hin_2.wav": (HI_V3[3], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hindi_3.wav": (HI_V3[4], "hi"),
    "data/processed/amaanvoice/hindivoice/aman_hin_4.wav": (HI_V3[5] + " " + HI_V3[6], "hi"),
    "data/processed/amaanvoice/hindivoice/Amman_hin_5.wav": (HI_V3[7] + " " + HI_V3[8], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin _6.wav": (HI_V3[9], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin _7 2.wav": (HI_V3[10] + " " + HI_V3[11], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin_8.wav": (HI_V3[12] + " " + HI_V3[13], "hi"),
    
    # --- HINDI v2 batch (hin_9 through hin_14) ---
    "data/processed/amaanvoice/hindivoice/aman_hin_9.wav": (HI_V2[1] + " " + HI_V2[2], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin_10.wav": (HI_V2[3] + " " + HI_V2[4], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin_11.wav": (HI_V2[5] + " " + HI_V2[6], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin_12.wav": (HI_V2[7] + " " + HI_V2[8], "hi"),
    "data/processed/amaanvoice/hindivoice/aman_hin_13.wav": (HI_V2[9], "hi"),
    "data/processed/amaanvoice/hindivoice/amaan_hin_14.wav": (HI_V2[10] + " " + HI_V2[11] + " " + HI_V2[12], "hi"),
}

# ============================================================
# STEP 3: Write the fixed CSV
# ============================================================
metadata_path = PROJECT_ROOT / "data/metadata/metadata.csv"

# Read existing to preserve order
with open(metadata_path, "r", encoding="utf-8") as f:
    existing_rows = list(csv.DictReader(f))

fixed_rows = []
fixed_count = 0
for row in existing_rows:
    path = row["audio_path"]
    if path in correct_labels:
        new_transcript, lang = correct_labels[path]
        old_t = row["transcript"]
        if old_t != new_transcript:
            fixed_count += 1
        row["transcript"] = new_transcript
        row["language"] = lang
    fixed_rows.append(row)

# Write fixed CSV
with open(metadata_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "language", "speaker"])
    writer.writeheader()
    writer.writerows(fixed_rows)

print(f"✅ Fixed {fixed_count} labels out of {len(fixed_rows)} total files.")
print(f"📁 Saved to: {metadata_path}")

# Verify - count by language
en_c = sum(1 for r in fixed_rows if r["language"] == "en")
hi_c = sum(1 for r in fixed_rows if r["language"] == "hi")
gu_c = sum(1 for r in fixed_rows if r["language"] == "gu")
print(f"\n📊 Language breakdown:")
print(f"  English:  {en_c}")
print(f"  Hindi:    {hi_c}")
print(f"  Gujarati: {gu_c}")
print(f"  Total:    {len(fixed_rows)}")
