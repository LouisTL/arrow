# Langage Arrow

Arrow est un petit langage de programmation créé pour le fun. Le premier interpréteur et compilateur du langage sont écrits en Python et le langage a aussi un compilateur écrit dans le langage lui-même. Exemple de code : 
```
fn fib(n: int): int {
    if (n <= 0) {
        return 0;
    } else if (n = 1){
        return 1;
    }

    nums <- [0,1];
    i <- 2;
    while (i <= n){
        push(nums, nums[i-1] + nums[i-2]);
        i <- i + 1;
    }
    return nums[n];
}

x: int <- fib(6);
print(x);
```
Comme vous pouvez voir dans l'exemple, <- est utilisé pour assigner à des variables et c'est de là que vient le nom du langage.

## Fonctionnement général
Pour le moment, Arrow possède trois manières d'exécuter des programmes. 
1. La première est d'utiliser un interpréteur écrit en Python dans lang.py. En ligne de commande, on peut exécuter ceci pour rouler un programme Arrow :
    ```
    python lang.py fichier.arrow
    ```
2. La deuxième manière est d'utiliser compiler.py. Ce fichier Python est un compilateur qui transforme le code Arrow en LLVM IR à l'aide de llvmlite. Ce compilateur était un peu pour tester le compilateur écrit en Arrow et il n'est plus beaucoup utilisé. Voici quand même comment l'utiliser en ligne de commande :
    ```
    python compiler.py fichier.arrow -o test.exe
    .\test.exe
    ```
    ou encore, en utilisant la Just-In-Time compilation de LLVM :
    ```
    python compiler.py fichier.arrow
    ```
3. La dernière façon et celle à privilégier, c'est d'utiliser le compilateur écrit en Arrow. C'est possible de l'invoquer avec Python comme ceci : 
    ```
    python lang.py compiler.arrow fichier.arrow -o test.exe
    .\test.exe
    ```
    Ce qui est intéressant, c'est qu'on peut compiler le compilateur lui-même en faisant comme ceci : 
    ```
    python lang.py compiler.arrow compiler.arrow -o arrow.exe
    ```
    C'est exactement comme ça que l'exécutable arrow.exe à la racine du repo a été généré. Vous pouvez réutiliser l'exécutable Windows (arrow.exe) ou l'exécutable Linux (arrow) ou bien recompiler le compilateur vous-mêmes. Une fois que vous avez l'exécutable, vous pouvez exécuter des programmes Arrow comme ceci :
    ```
    arrow.exe fichier.arrow
    .\fichier.exe
    ```
    Comme compiler.py, compiler.arrow transforme aussi le code Arrow en LLVM IR. Ce qui diffère par contre, c'est que pour le moment compiler.arrow génère un fichier .ll temporaire avec le IR de LLVM écrit en texte. Ensuite, une commande shell est exécutée pour appeler clang qui va optimiser le code IR et le compiler en langage machine. Le fichier .ll est supprimé automatiquement après que notre backend, c'est-à-dire clang, complète la compilation.
    
    Fait intéressant à noter : le langage a la "fixed-point compilation". Cela veut dire qu'on peut recompiler le compilateur autant de fois qu'on veut et que le fichier .ll généré sera identique. On peut générer un deuxième compilateur natif à partir du premier en exécutant ceci : 
    ```
    arrow.exe compiler.arrow -o arrow2.exe
    ```

Pour vérifier que tout fonctionne, vous pouvez lancer `python run_tests.py` pour exécuter tous les exemples avec l'interpréteur et le compilateur natif et comparer les sorties. Il y a quelques différences connues en ce moment, principalement liées au formatage de print() pour les types composés.
 ## Dépendances
- Python 3.10+
- Clang 15+
## Fonctionnalités
Le langage a un nombre limité de fonctionnalités pour le moment, mais ce nombre risque d'augmenter dans le futur.
### types
Arrow supporte les types suivants :
int, string, struct, array et function. Voici un exemple de définition pour chacun:
```
num <- 42;
hello <- "world";
token <- {type: "struct", nom: "token"};
liste <- [1,2,3];
fleche <- (x) => {return x+1;};
```
Il y a certaines règles de typage qui existent. Par exemple, la compilation va échouer si on tente de passer un int à une fn explicitement typée pour accepter un string. Cela étant dit, les règles de typage ne sont pas très strictes à ce moment-ci et les annotations de type ne sont pas obligatoires.

Le système de types supporte aussi les types structurés comme ceux-ci :
```
liste: [string] <- ["alpha", "bravo", "charlie"];
point: {x: int, y: int} <- {x: 3, y: 4};
```
### instructions
Quelques instructions de base sont supportées, comme celles-ci : 
```
if (val = autre){
    print("Les valeurs sont égales");
} else {
    print("Les valeurs ne sont pas égales");
}

while (x < y){
    print("boucle while");
}

for (char in "string") {
    print(char);
}

for (char in [1,2,3]) {
    print(char);
}
```
### fonctions de base
Des fonctions de base existent dépendamment des types utilisés. Voici une petite liste de ces fonctions :
- len() peut être utilisé pour des arrays ou des strings
- push() et pop() peuvent être utilisés pour ajouter ou retirer des éléments dans des arrays
- char_code(), from_char_code(), substring() et char_at() existent pour les string
- keys() peut être utilisé pour obtenir les clés de struct
### I/O
Le langage supporte read_file pour lire un fichier, write_file pour écrire dans un fichier, append_file pour ajouter à un fichier et input() pour lire stdin (comme input() en Python).
### examples
Il y a beaucoup d'exemples de code dans le répertoire examples du repo. Vous pouvez les consulter pour plus vous familiariser avec Arrow.
### fonctionnalités manquantes
La liste est longue, beaucoup plus longue que la liste de fonctionnalités présentes. Il n'y a pas encore de floats, de OOP (pas certain que je vais l'ajouter par contre), de généricité, de librairie standard, de dates, de structures de données comme des dictionnaires, de modules/imports, etc.