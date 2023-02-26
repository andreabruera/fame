class ExperimentOne:

    def __init__(self):

        self.presentation = 'Benvenuto e grazie per il tuo contributo!'\
                            '\n\n'\
                            'Nel corso dell\'esperimento, appariranno prima delle parole da sole, riferite a persone e luoghi famosi o a te familiari.'\
                            '\n\n'\
                            'Quando le parole appariranno, ti chiediamo di pensare a ciò a cui si riferiscono e alle sue catteristiche,'\
                            ' come se dovessi immaginartelo nella tua testa.'\
                            '\n\n'\
                            'Dopodichè, apparirà sullo schermo una domanda riguardante il significato di quelle parole, a cui dovrai cercare di rispondere '\
                            'con attenzione e correttamente, usando i tasti indicati sullo schermo.'\
                            '\n\n\n'\
                            '[Premi la barra spaziatrice per continuare]'
                            
        self.organization = 'L\'esperimento è composto da 24 brevi sessioni da circa un minuto e mezzo, intervallate da pause di 30 secondi.'\
                            '\n\n'\
                            'Durante tutto l\'esperimento, prima e dopo l\'apparizione delle parole, verrà proiettata una croce in centro allo'\
                            ' schermo, che ti chiediamo di fissare.'\
                            '\n\n\n'\
                            '[Premi la barra spaziatrice per continuare]'
                            
        self.recommendations = 'Ti preghiamo, al fine di non rovinare i dati registrati, di sbattere gli occhi il meno possibile durante '\
                               'le sessioni sperimentali - e di non muoverti, per lo stesso motivo.'\
                               '\n\n'\
                               'Durante le pause, invece, potrai rilassarti e riposare gli occhi.'\
                               '\n\n'\
                               'E quindi... ora è il momento di metterti a tuo agio - per quanto possibile, con tutti quegli elettrodi :) - '\
                               'e cercare una posizione comoda, e poi possiamo cominciare!'\
                               '\n\n\n'\
                               '[Premi la barra spaziatrice quando vuoi proseguire]'
                               
        self.start = 'Ci siamo!\n\n\n[Premi la barra spaziatrice per cominciare]'
        
        self.break_message = lambda values : 'Hai finito la sessione {} su 24.\n\nLa tua percentuale di risposte esatte '\
                                             'è stata del {}%.'\
                                             '\n\n'\
                                             'Secondi rimanenti prima della prossima sessione: {}'\
                                             ''.format(int(values[0])+1, values[1], values[2])
                                             
        self.end = 'Finito! Grazie ancora per la partecipazione.'
