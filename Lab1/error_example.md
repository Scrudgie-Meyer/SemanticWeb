# Помилковий та виправлений приклад

## Помилковий варіант

```turtle
cars:ToyotaCorolla cars:manufacturedBy "Toyota" .
```

Об'єктом є текстовий літерал. Через це неможливо перейти від автомобіля до сутності виробника, а потім до країни його походження.

## Виправлений варіант

```turtle
cars:ToyotaCorolla cars:manufacturedBy cars:Toyota .
cars:Toyota a cars:Manufacturer ;
    schema:name "Toyota" ;
    cars:countryOfOrigin cars:Japan .
```

Тепер виробник є окремим RDF-ресурсом із власним URI. Це дає змогу виконувати запити з переходом між автомобілем, виробником, країною та частиною світу.
