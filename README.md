# Odporúčací systém s využitím umelej inteligencie

Python aplikácia na generovanie presonalizovaných odporúčaní s webovým rozhraním, pre trénovanie modelov bol použitý framework TensorFlow.
Testované na Windows 10, Python 3.12.3, TensorFlow 2.16.

## Návod na použitie 

Kliknutím na symbol srdca označíte, že sa vám obrázok páči, po 5 označeniach môžete kliknúť na tlačidlo Odporúčiť, vtedy sa vyšle žiadosť na server a ten vytrénuje model podlá vašich preferencií a vráti zoznam relevantných obrázkov, kde na vrchu sa zobrazujú najrelevantnejšie obrázky a naspodku najmenej relevantné.
Ďalej sa zobrazí tlačidlo, kde si môžete vyberať spôsob odporúčania Content - Na základe obsahu, Collaborative - Porovnávanie s používateľmi, Hybrid - Zmiešané, Liked - Vaše označené obrázky.


## Lokálne spustenie / GITHUB

1. Stiahnutie potrebných knižníc a frameworkov

    ```
    pip install -r requirements.txt
    ```

2. Stiahnutie obrázkov 

    Kvôli veľkosti obrázkov potrebných na projekt sú uložené na google disku.
    Stiahnite si priečinky z nasledujúceho linku a vložte ich do adresára "projekt".
    ```
    https://docs.docker.com/desktop/install/windows-install/
    ```
   
3. Spustenie Flask serveru

    ```
    flask --app app run
    ```

4. Navštívte lokálnu IP adresu flask serveru

    ```
    http://127.0.0.1:5000
    ```

## Lokálne spustenie / DOCKER

Docker image je už zbuildovaný stačí ho len stiahnuť (pull) a spustiť (run).

1. Je potrebné mať setupnutý docker 

    ```
    https://docs.docker.com/desktop/install/windows-install/
    ```

2. Stiahnite si image

    ```
    docker pull xbercikf/bcpraca
    ```

3. Spustite image

    ```
    docker run -d -p 5000:5000 xbercikf/bcpraca
    ```
    
4. Navštívte lokálnu IP adresu flask serveru

    ```
    localhost:5000
    ```


## Evaluácia modelov

V priečinkoch /projekt/show nájdete súbor showEval.py, ktorý hodnotí úspešnosť trénovaních modelov na vzorových použivateľoch.

## Kategorizácia obrázkov

Modely happy, minimalistic, modern, realistic majú každý svoj priečinok kde je rovnomenný súbor zdrojového kódu Python, ten trénuje modely a ukladá ich pod menom xyzModel.keras, kde xyz je meno kategórie, podľa ktorej obrázky zaraďuje.
Súbor categorizeData.py vezme tieto modely a kategorizuje všetky obrázky z base priečinku a vloží tieto dáta do baseCategorized.xlsx excel súboru.