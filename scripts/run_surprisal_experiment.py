'''
Main orchestrator for the surprisal experiment. 

1. Load code switched words from CSV
2. Convert code switch -> full cantonese
3. Find matching monolingual sentences for surprisal comparison
4. Calculate surprisal for both original code switch word & monolingual word (at the code switch pos)
5. Compare surprisal values 
6. Log into a CSV file & creates figures 

'''