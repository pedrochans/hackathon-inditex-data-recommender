import pandas as pd

def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be scrictly followed.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """
    
    # Filtramos los datos para el usuario
    user_data = df[df['user_id'] == user_id]
    user_data.loc[:,'timestamp_local'] = pd.to_datetime(user_data['timestamp_local'])

    # Si no hay datos para el usuario, devolvemos un DataFrame vacío
    if user_data.empty:
        return pd.DataFrame(columns=['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio'])

    session_metrics_df = []
    
    for session_id in user_data['session_id'].unique():
        session_data = user_data[user_data['session_id'] == session_id]
        
        # Calculamos el tiempo total de la sesión
        total_session_time = (session_data['timestamp_local'].max() - session_data['timestamp_local'].min()).total_seconds()
        total_session_time = round(total_session_time, 2)
        
        '''
        # Calculamos la proporción de productos añadidos al carrito sobre productos interaccionados
        total_products_cart = session_data[session_data['add_to_cart']==1]['partnumber'].nunique()
        total_products_visited = session_data['partnumber'].nunique()
        cart_addition_ratio = round(100 * total_products_cart / total_products_visited if total_products_visited > 0 else 0, 2)
        '''

        # Calculamos la proporción de adiciones al carrito sobre interaccionados
        total_products_cart = session_data['add_to_cart'].sum()
        total_products_visited = len(session_data)
        cart_addition_ratio = round(100 * total_products_cart / total_products_visited if total_products_visited > 0 else 0, 2)

        session_metrics_df.append({
            'user_id': user_id,
            'session_id': session_id,
            'total_session_time': total_session_time,
            'cart_addition_ratio': cart_addition_ratio
        })
    
    return pd.DataFrame(session_metrics_df).sort_values('session_id', ascending=True, ignore_index=True)

