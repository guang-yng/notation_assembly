import csv

# Input data string
data_str = """
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 
                   all         28      18330      0.657      0.285       0.47      0.378
          noteheadFull         28       4376      0.996      0.449      0.723      0.525
                  stem         28       4382      0.998      0.212      0.605      0.394
                  beam         28       1249      0.992      0.491      0.743      0.561
       augmentationDot         28        445      0.933     0.0315      0.482      0.231
       accidentalSharp         28        402      0.994      0.413      0.705      0.611
        accidentalFlat         28        249      0.987      0.627      0.807      0.666
     accidentalNatural         28        220      0.981      0.714      0.854      0.707
 accidentalDoubleSharp         28          9          0          0          0          0
             restWhole         28         30          1      0.367      0.683      0.589
              restHalf         28         59          1      0.271      0.636      0.525
           restQuarter         28        168          1      0.595      0.798      0.731
               rest8th         28        204          1       0.73      0.865      0.775
              rest16th         28         51      0.974      0.745      0.868      0.767
      multiMeasureRest         28         12      0.923          1      0.976      0.826
             legerLine         28       1275      0.912     0.0816      0.496       0.24
 graceNoteAcciaccatura         28         12      0.857        0.5      0.714      0.414
     noteheadFullSmall         28         58          1     0.0345      0.517      0.155
               bracket         28          8          0          0          0          0
                 brace         28         25      0.933       0.56      0.752      0.602
               barline         28        637      0.993      0.455      0.726      0.595
          barlineHeavy         28          9          0          0          0          0
      measureSeparator         28        555      0.913     0.0378      0.476      0.364
                repeat         28         15          1        0.4        0.7       0.57
             repeatDot         28         32          0          0          0          0
  articulationStaccato         28        205          0          0          0          0
          characterDot         28         36          0          0          0          0
    articulationTenuto         28          7          0          0          0          0
    articulationAccent         28         56          1     0.0179      0.509      0.407
                  slur         28        621      0.926      0.441      0.688      0.613
                   tie         28        120       0.93      0.333      0.634      0.528
dynamicCrescendoHairpin         28         54      0.913      0.389      0.661      0.581
dynamicDiminuendoHairpin         28         88          1     0.0909      0.545      0.463
              ornament         28         10        0.9        0.9      0.944      0.867
           wiggleTrill         28          3       0.75          1      0.746      0.481
         ornamentTrill         28         33      0.938      0.455       0.71      0.611
              arpeggio         28          5          0          0          0          0
             glissando         28          4          1          1      0.995      0.814
   multipleNoteTremolo         28         32          0          0          0          0
          tupleBracket         28         15      0.889      0.533      0.733      0.561
                 tuple         28         32          1      0.281      0.641      0.484
                 gClef         28         76          1      0.474      0.737      0.704
                 fClef         28         66          1      0.697      0.848      0.803
                 cClef         28         35          1      0.914      0.957       0.92
          keySignature         28        125      0.982      0.432       0.71      0.655
         timeSignature         28         35      0.955        0.6      0.788      0.704
         timeSigCommon         28          7          1      0.429      0.714      0.643
          dynamicsText         28        152          1      0.296      0.648      0.576
             tempoText         28         24      0.923        0.5      0.727      0.643
             otherText         28         49      0.833      0.612      0.731      0.541
       characterSmallA         28         15          1     0.0667      0.533      0.267
       characterSmallB         28          2          0          0          0          0
       characterSmallC         28         50      0.667       0.12      0.385      0.279
       characterSmallD         28         35       0.75     0.0857      0.421      0.282
       characterSmallE         28         78      0.857     0.0769      0.471      0.286
       characterSmallF         28        104          1     0.0192       0.51      0.484
       characterSmallG         28          3          0          0          0          0
       characterSmallI         28         18          0          0          0          0
       characterSmallJ         28          2          0          0          0          0
       characterSmallL         28         13          0          0          0          0
       characterSmallM         28         36        0.8      0.111      0.444       0.28
       characterSmallN         28         13          0          0          0          0
       characterSmallO         28         67      0.938      0.224      0.588      0.423
       characterSmallP         28        104      0.917      0.106      0.512      0.407
       characterSmallR         28         70          1     0.0714      0.536      0.343
       characterSmallS         28         58          0          0          0          0
       characterSmallT         28         53          1      0.132      0.566      0.438
       characterSmallU         28         11          0          0          0          0
       characterSmallV         28          1          0          0          0          0
       characterSmallZ         28          3          0          0          0          0
     characterCapitalA         28          1          1          1      0.995      0.895
     characterCapitalF         28          8          0          0          0          0
     characterCapitalM         28          6      0.333      0.333      0.278      0.234
     characterCapitalP         28         17      0.833      0.294      0.539      0.422
     characterCapitalT         28          3       0.75          1      0.746      0.655
     characterCapitalV         28          1          0          0          0          0
              numeral2         28          2          0          0          0          0
              numeral3         28         46      0.933      0.304      0.628      0.554
              numeral4         28         22          1      0.591      0.795      0.708
              numeral5         28          2          0          0          0          0
              numeral6         28          6          1        0.5       0.75      0.569
              numeral7         28          4          0          0          0          0
              numeral8         28          7          1      0.429      0.714      0.614
    instrumentSpecific         28          4          1       0.25      0.625      0.562
          unclassified         28         11          0          0          0          0
dottedHorizontalSpanner         28          3          0          0          0          0
     transpositionText         28          1          0          0          0          0
        characterOther         28          6          1        0.5       0.75      0.492
            breathMark         28          2          0          0          0          0
          noteheadHalf         28        372      0.992       0.32      0.657      0.512
         noteheadWhole         28         54          1      0.444      0.722      0.607
             flag8thUp         28        128      0.968      0.469      0.721      0.528
           flag8thDown         28        222      0.991      0.509      0.749      0.563
            flag16thUp         28         28          1     0.0714      0.536      0.347
          flag16thDown         28         31          0          0          0          0
          fermataAbove         28          8          1       0.75      0.875      0.779
        dynamicLetterP         28         82          1     0.0854      0.543      0.466
        dynamicLetterM         28         15          0          0          0          0
        dynamicLetterF         28        112          1      0.179      0.589      0.483
        dynamicLetterS         28         18          1     0.0556      0.528      0.475
"""

# Parsing the data string
# Splitting the data into lines and then into values
lines = data_str.strip().split('\n')
data = [line.split() for line in lines]

# Defining the CSV file name
csv_file_path = 'parsed_res.csv'

# Writing data to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Optional: Write headers (if you know what each column represents)
    # headers = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6', 'Column7']
    # writer.writerow(headers)
    for row in data:
        writer.writerow(row)


