# Example prior knowledge for data/example.duckdb (optional; copy to sources/example/context.md)

## Columns
grain_quotes.price_mid is the mid price in the row's currency per tonne; price_low and price_high are the band around it.
grain_quotes.basis says whether a row is an observed quote or an outlook (forecast) revised later.

## Names
Series codes such as WHT_MILL_GP name one crop at one region, e.g. milling wheat in the Great Plains.
Bulletins tag crops and regions by free text, so spellings can differ from the quotes tables.

## Currencies
Prices are quoted in USD, except European series (e.g. rapeseed) which are in EUR per tonne.

## News
farm_bulletins are weekly agricultural bulletins; they explain supply moves such as the spring 2025 drought.
