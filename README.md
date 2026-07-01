# Langage Arrow

Arrow est un petit langage de programmation créé pour le fun. Le premier interpréteur et compilateur du langage sont écrits en Python et le langage a aussi un compilateur écrit dans le langage lui-même. Exemple de code : 
```
fn fib(n: int): int {
    if (n <= 0) {
        return 0;
    } else if (n = 1){
        return 1;
    }

    var nums <- [0,1];
    var i <- 2;
    while (i <= n){
        push(nums, nums[i-1] + nums[i-2]);
        i <- i + 1;
    }
    return nums[n];
}

var x: int <- fib(6);
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
    Comme compiler.py, compiler.arrow transforme aussi le code Arrow en LLVM IR. Ce qui diffère par contre, c'est que pour le moment compiler.arrow génère un fichier .ll temporaire avec le IR de LLVM écrit en texte. Ensuite, une commande shell est exécutée pour appeler clang qui va optimiser le code IR et le compiler en langage machine. Le fichier .ll est supprimé automatiquement après que notre backend, c'est-à-dire clang, complète la compilation (on peut le conserver avec le flag `--keep-ll`).
    
    Fait intéressant à noter : le langage a la "fixed-point compilation". Cela veut dire qu'on peut recompiler le compilateur autant de fois qu'on veut et que le fichier .ll généré sera identique. On peut générer un deuxième compilateur natif à partir du premier en exécutant ceci : 
    ```
    arrow.exe compiler.arrow -o arrow2.exe
    ```

Pour vérifier que tout fonctionne, vous pouvez lancer `python run_tests.py` pour exécuter tous les exemples avec l'interpréteur et le compilateur natif et comparer les sorties. Présentement, tous les exemples passent.
 ## Dépendances
- Python 3.10+
- Clang 15+
## Fonctionnalités
Le langage a un nombre limité de fonctionnalités pour le moment, mais ce nombre risque d'augmenter dans le futur.
### types
Arrow supporte les types suivants :
int, float, str, bool, struct, array, union et function (plus `any`, une échappatoire au typage). Voici un exemple de définition pour quelques-uns :
```
var num <- 42;
var hello <- "world";
var token <- {type: "struct", nom: "token"};
var liste <- [1,2,3];
var fleche <- (x) => {return x+1;};
```
Il y a certaines règles de typage qui existent. Par exemple, la compilation va échouer si on tente de passer un int à une fn explicitement typée pour accepter un str. Cela étant dit, les règles de typage ne sont pas très strictes à ce moment-ci et les annotations de type ne sont pas obligatoires.

Le système de types supporte aussi les types structurés comme ceux-ci :
```
var liste: [str] <- ["alpha", "bravo", "charlie"];
var point: {x: int, y: int} <- {x: 3, y: 4};
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
### floats
Arrow supporte les nombres à virgule flottante (`float`). Un littéral avec un point décimal est un float, et l'arithmétique fait la promotion automatiquement : si l'un des deux opérandes est un float, le résultat l'est aussi.
```
var pi <- 3.14;
var moitie <- 10.0 / 4.0;       // 2.5
print(1 + 1.5);                 // 2.5   (int promu en float)
print(10 / 3.0);                // 3.3333333333333335

var xs: [float] <- [0.5, 1.5, 2.5];
print(xs[1]);                   // 1.5
```
L'affichage suit le même format que le `repr` de Python : la représentation décimale la plus courte qui fait un aller-retour exact, avec un ".0" final pour les floats de valeur entière.
### unions et pattern matching
Un type union `A | B` accepte une valeur de l'un ou l'autre type. On inspecte une union avec `match`, qui vérifie l'exhaustivité à la compilation : chaque membre doit être couvert par un bras typé ou par le bras joker `_`.
```
fn describe(v: int | str | bool) {
    match (v) {
        42  => { print("le nombre 42"); }   // motif littéral (raffine int)
        int => { print("un autre int"); }   // bras typé, lie v (int)
        _   => { print(v); }                 // joker : couvre str et bool
    }
}
describe(42);       // le nombre 42
describe(7);        // un autre int
describe("allo");   // allo
describe(true);     // true
```
`match` s'utilise aussi comme expression qui retourne une valeur :
```
fn nom(v: int | str): str {
    return match (v) {
        int => "c'est un entier",
        str => "c'est une chaine"
    };
}
```
Un bras littéral (comme `42`) raffine un membre mais ne compte pas pour l'exhaustivité; un bras typé ou `_` doit quand même couvrir chaque membre. Le bras joker `_`, s'il est présent, doit être le dernier.
### alias de types
On peut nommer un type structurel avec `type Nom <- Type;`. L'alias se résout à sa structure sous-jacente (le typage demeure structurel) :
```
type Point <- {x: int, y: int};
type Shape <- {radius: int} | {side: int};
type IntList <- [int];

var p: Point <- {x: 1, y: 2};
print(p.x);                     // 1
```
Un nom de type inconnu est une erreur de compilation
### modules et imports
Un fichier peut en importer un autre avec `import "chemin";`. Chaque fonction et variable globale du fichier importé devient accessible via `chemin.<nom>`, où l'espace de noms est le nom de base du fichier (renommable avec `as`).
```
import "mathlib" as m;         // charge ./mathlib.arrow

print(m.square(7));            // 49
print(m.cube(3));              // 27
```
Le résolveur cherche d'abord le fichier relativement à celui qui l'importe, puis dans le répertoire de l'exécutable arrow (où se trouve `std/`). C'est ce qui permet à `import "std/math"` de fonctionner de n'importe où (voir la librairie standard ci-dessous).
### librairie standard
La librairie standard ne fait que commencer, avec le module `std/math` (`abs`, `min`, `max`) :
```
import "std/math";

print(math.abs(-5));           // 5
print(math.min(3, 8));         // 3
print(math.max(3, 8));         // 8
```
Elle grandira à mesure que le langage gagnera en fonctionnalités.
### examples
Il y a beaucoup d'exemples de code dans le répertoire examples du repo. Vous pouvez les consulter pour plus vous familiariser avec Arrow.
### fonctionnalités manquantes
Plusieurs fonctionnalités restent à ajouter. Il n'y a pas encore de OOP (pas certain que je vais l'ajouter par contre), de généricité, de dates, ni de structures de données comme des dictionnaires, etc.
