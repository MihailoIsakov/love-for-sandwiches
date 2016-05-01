#-*-coding:utf-8-*-
#
#    Simple stemmer for Croatian v0.1
#    Copyright 2012 Nikola Ljubešić and Ivan Pandžić
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import re
import sys
stop=set(['biti','jesam','budem','sam','jesi','budeš','si','jesmo','budemo','smo','jeste','budete','ste','jesu','budu','su','bih','bijah','bjeh','bijaše','bi','bje','bješe','bijasmo','bismo','bjesmo','bijaste','biste','bjeste','bijahu','biste','bjeste','bijahu','bi','biše','bjehu','bješe','bio','bili','budimo','budite','bila','bilo','bile','ću','ćeš','će','ćemo','ćete','želim','želiš','želi','želimo','želite','žele','moram','moraš','mora','moramo','morate','moraju','trebam','trebaš','treba','trebamo','trebate','trebaju','mogu','možeš','može','možemo','možete'])


def istakniSlogotvornoR(niz):
    return re.sub(r'(^|[^aeiou])r($|[^aeiou])',r'\1R\2',niz)

def imaSamoglasnik(niz):
    if re.search(r'[aeiouR]',istakniSlogotvornoR(niz)) is None:
        return False
    else:
        return True

def transformiraj(pojavnica):
    for trazi,zamijeni in transformacije:
        if pojavnica.endswith(trazi):
            return pojavnica[:-len(trazi)]+zamijeni
    return pojavnica

def korjenuj(pojavnica):
    for pravilo in pravila:
        dioba=pravilo.match(pojavnica)
        if dioba is not None:
            if imaSamoglasnik(dioba.group(1)) and len(dioba.group(1))>1:
                return dioba.group(1)
    return pojavnica


def newline2omega(text):
    return text.replace(u'\n', u' Ω')


def omega2newline(text):
    return text.replace(u'Ω', '\n')

    
def main():
    global transformacije, pravila
    if len(sys.argv)!=3:
        print 'Usage: python Croatian_stemmer.py input_file output_file'
        print 'input_file should be an utf8-encoded text file which is then tokenized, stemmed and written in the output_file in a tab-separated fashion.'
        sys.exit(1)

    output_file=open(sys.argv[2],'w')
    pravila=[re.compile(r'^('+osnova+')('+nastavak+r')$') for osnova, nastavak in [e.decode('utf8').strip().split(' ') for e in open('rules.txt')]]
    transformacije=[e.decode('utf8').strip().split('\t') for e in open('transformations.txt')]

    input_file = open(sys.argv[1])
    for line in input_file:
        line = line.decode('utf8')
        line = newline2omega(line)

        for token in re.findall(r'\w+', line, re.UNICODE):
            token = omega2newline(token.strip())
            if token.lower() in stop:
                output_file.write((token.lower() + ' ').encode('utf8'))
            else:
                output_file.write((korjenuj(transformiraj(token.lower()))+' ').encode('utf8'))

    output_file.close()

if __name__=='__main__':
    main()

