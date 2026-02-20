# estimation de la tendance
x_num = np.arange(len(spy_vol.dropna()))  # Index numérique pour régression
y_clean = spy_vol.dropna().values

coeff = np.polyfit(x_num, y_clean, 100)  # régression par polynome de degré 100
p_trend = np.poly1d(coeff)
y_trend = p_trend(x_num)

def plot_data_trend(x, y, x_trend, y_trend, mode, name_data, name_trend, title, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name_data))
    fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name=name_trend, line=dict(color='red', width=3)))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, hovermode="x unified")
    fig.show()
    return 0

# Appel
plot_data_trend(spy_vol.index[1:], y_clean, spy_vol.index[1:], y_trend, 'lines+markers', 'Differenced Volume', 'Tendance linéaire', "SPY Volume Diff + Tendance", "Datetime", "Volume Diff")

# Moyenne mobile en prenant en compte la saisonnalité
# Période saisonnière (ex: 26 périodes 15min = 6h30, 1/2 session trading)
period_saison = 26  

# Moyenne mobile saisonnière (sur 1 cycle complet)
vol_saison = spy_vol.rolling(window=period_saison, center=True).mean()

# Composante saisonnière = moyenne mobile sur la période

# Données désaisonnalisées
vol_desaisson = spy_vol - vol_saison

# Visualisation avec votre fonction
plot_data(spy_vol.index, vol_saison.values, 'lines', 'Saisonnalité', 
           "SPY Volume - Composante Saisonnière (MM 26p)", "Datetime", "Volume")
           
plot_data(spy_vol.index, vol_desaisson.values, 'lines', 'Désaisonnalisé', 
           "SPY Volume - Données Désaisonnalisées", "Datetime", "Volume Désaison.")

