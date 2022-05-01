damp = 0.8
Imperial_Spice = 3
Shahi_Darbar = 2
CardamoMughlai_Darbaramom = 0
Mughlai_Darbar = 2
Indian_Grill = 2
King_Of_Spice = 0
Imperial_Spice_rank = 1
Shahi_Darbar_rank = 1
Cardamom_rank = 1
Mughlai_Darbar_rank = 1
Indian_Grill_rank = 1
King_Of_Spice_rank = 1
iteration = 0
while iteration < 10:
a = (1-damp)+(damp*(Shahi_Darbar_rank/Shahi_Darbar))
b = (1-damp)+(damp*(Imperial_Spice_rank/Imperial_Spice))
c = (1-damp)+(damp*(Shahi_Darbar_rank/Shahi_Darbar +
Mughlai_Darbar_rank/Mughlai_Darbar))
d = (1-damp)+damp*(Indian_Grill_rank/Indian_Grill)
e = (1-damp)+damp*(Imperial_Spice_rank /
Imperial_Spice+Mughlai_Darbar_rank/Mughlai_Darbar)
f = (1-damp)+damp*(Imperial_Spice_rank/Imperial_Spice)
Imperial_Spice_rank = a
Shahi_Darbar_rank = b
Cardamom_rank = c
Mughlai_Darbar_rank = d
Indian_Grill_rank = e
King_Of_Spice_rank = f
iteration += 1
print("Rank prestige value of Shahi Darbar is " ,Shahi_Darbar_rank)
print("Rank prestige value of Imperial Spice is ", Imperial_Spice_rank)
print("Rank prestige value of Mughlai Darbar is ", Mughlai_Darbar_rank)
print("Rank prestige value of Indian Grill is " ,Indian_Grill_rank)
print("Rank prestige value of King of Spices is ", King_Of_Spice_rank)
print("Rank prestige value of Cardamom is " ,Cardamom_rank)
