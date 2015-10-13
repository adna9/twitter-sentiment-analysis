def numberOfPronounVerbVerb(trigrams):
  return len([a for a,b,c in trigrams if a=="O" and b=="V" and c =="V"])

def numberOfVerbDeterminerNoun(trigrams):
  return len([a for a,b,c in trigrams if a=="V" and b=="D" and c =="N"])

def numberOfPositionDeterminerNoun(trigrams):
  return len([a for a,b,c in trigrams if a=="P" and b=="D" and c =="N"])
